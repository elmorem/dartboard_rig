"""
FastAPI routes for Dartboard RAG API.

Endpoints:
- POST /query: Answer questions using RAG
- POST /ingest: Ingest documents into vector store
- GET /health: Health check and system status
"""

import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware

from dartboard.api.models import (
    QueryRequest,
    QueryResponse,
    SourceModel,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    ErrorResponse,
)
from dartboard.api.dependencies import (
    get_hybrid_retriever,
    get_rag_generator,
    get_ingestion_pipeline,
    get_vector_store,
    get_config,
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Dartboard RAG API",
    description="Retrieval-Augmented Generation with Dartboard algorithm",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    retriever=Depends(get_hybrid_retriever),
    generator=Depends(get_rag_generator),
):
    """
    Answer a question using RAG.

    Process:
    1. Retrieve relevant chunks using hybrid retrieval (vector + Dartboard)
    2. Generate answer using LLM with source citations

    Args:
        request: Query request with question and parameters
        retriever: Hybrid retriever (injected)
        generator: RAG generator (injected)

    Returns:
        QueryResponse with answer, sources, and timing information
    """
    try:
        start_time = time.time()

        # Step 1: Retrieve relevant chunks
        retrieval_start = time.time()
        logger.info(f"Retrieving chunks for query: {request.query[:100]}")

        retrieval_result = retriever.retrieve(
            query=request.query, k=request.top_k, sigma=request.sigma
        )

        retrieval_time = (time.time() - retrieval_start) * 1000  # ms

        if not retrieval_result.chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found in vector store",
            )

        # Step 2: Generate answer
        generation_start = time.time()
        logger.info(f"Generating answer with {len(retrieval_result.chunks)} chunks")

        generation_result = generator.generate(
            query=request.query,
            chunks=retrieval_result.chunks,
            temperature=request.temperature,
        )

        generation_time = (time.time() - generation_start) * 1000  # ms
        total_time = (time.time() - start_time) * 1000  # ms

        # Build response
        sources = [
            SourceModel(
                number=source["number"],
                text=source["text"],
                metadata=source["metadata"],
                chunk_id=source.get("metadata", {}).get("source", "unknown"),
            )
            for source in generation_result.sources
        ]

        response = QueryResponse(
            answer=generation_result.answer,
            sources=sources,
            num_sources_cited=generation_result.num_sources_cited,
            retrieval_time_ms=round(retrieval_time, 2),
            generation_time_ms=round(generation_time, 2),
            total_time_ms=round(total_time, 2),
            model=generation_result.model,
        )

        logger.info(
            f"Query completed in {total_time:.0f}ms "
            f"(retrieval: {retrieval_time:.0f}ms, "
            f"generation: {generation_time:.0f}ms)"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    chunk_size: int = Form(512, ge=100, le=2000, description="Max tokens per chunk"),
    overlap: int = Form(50, ge=0, le=500, description="Token overlap between chunks"),
    track_progress: bool = Form(False, description="Enable progress logging"),
    pipeline=Depends(get_ingestion_pipeline),
):
    """
    Ingest a document into the vector store.

    Process:
    1. Save uploaded file temporarily
    2. Load document
    3. Chunk document
    4. Generate embeddings
    5. Store in vector database

    Args:
        file: Uploaded document file
        chunk_size: Maximum tokens per chunk
        overlap: Token overlap between chunks
        track_progress: Enable progress logging
        pipeline: Ingestion pipeline (injected)

    Returns:
        IngestResponse with ingestion statistics
    """
    try:
        start_time = time.time()

        # Validate file type
        allowed_extensions = {".md", ".txt", ".pdf"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. "
                f"Allowed: {', '.join(allowed_extensions)}",
            )

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        logger.info(f"Ingesting file: {file.filename} ({len(content)} bytes)")

        # Update chunker configuration
        pipeline.chunker.chunk_size = chunk_size
        pipeline.chunker.overlap = overlap

        # Ingest document
        result = pipeline.ingest(temp_path, track_progress=track_progress)

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        processing_time = (time.time() - start_time) * 1000  # ms

        response = IngestResponse(
            status=result.status,
            documents_processed=result.documents_processed,
            chunks_created=result.chunks_created,
            chunks_stored=result.chunks_stored,
            processing_time_ms=round(processing_time, 2),
            errors=result.errors,
            metadata={
                **result.metadata,
                "filename": file.filename,
                "file_size": len(content),
            },
        )

        logger.info(
            f"Ingestion completed: {result.chunks_stored} chunks stored "
            f"in {processing_time:.0f}ms"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Document ingestion failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    vector_store=Depends(get_vector_store), config=Depends(get_config)
):
    """
    Health check endpoint.

    Returns:
        HealthResponse with system status and configuration
    """
    try:
        # Get vector store count
        try:
            count = vector_store.count()
        except:
            count = 0

        response = HealthResponse(
            status="healthy",
            vector_store_count=count,
            embedding_model=config["embedding_model"],
            llm_model=config["llm_model"],
            version="1.0.0",
        )

        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Dartboard RAG API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation with Dartboard algorithm",
        "endpoints": {
            "query": "POST /query - Answer questions using RAG",
            "ingest": "POST /ingest - Ingest documents",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation",
        },
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Not Found", detail=str(exc), status_code=404
        ).model_dump(),
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            status_code=500,
        ).model_dump(),
    )
