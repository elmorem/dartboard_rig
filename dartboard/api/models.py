"""
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for /query endpoint."""

    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    sigma: float = Field(1.0, gt=0, description="Dartboard temperature parameter")
    temperature: float = Field(
        0.7, ge=0, le=2, description="LLM generation temperature"
    )


class SourceModel(BaseModel):
    """Source document in response."""

    number: int = Field(..., description="Source citation number")
    text: str = Field(..., description="Source text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Source metadata"
    )
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""

    answer: str = Field(..., description="Generated answer")
    sources: List[SourceModel] = Field(
        default_factory=list, description="Cited sources"
    )
    num_sources_cited: int = Field(..., description="Number of sources cited")
    retrieval_time_ms: float = Field(..., description="Time spent on retrieval (ms)")
    generation_time_ms: float = Field(..., description="Time spent on generation (ms)")
    total_time_ms: float = Field(..., description="Total processing time (ms)")
    model: str = Field(..., description="LLM model used")


class IngestRequest(BaseModel):
    """Request model for /ingest endpoint (metadata)."""

    chunk_size: int = Field(512, ge=100, le=2000, description="Max tokens per chunk")
    overlap: int = Field(50, ge=0, le=500, description="Token overlap between chunks")
    track_progress: bool = Field(False, description="Enable progress logging")


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint."""

    status: str = Field(..., description="Ingestion status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    chunks_stored: int = Field(..., description="Number of chunks stored in vector DB")
    processing_time_ms: float = Field(..., description="Processing time (ms)")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str = Field(..., description="Service status")
    vector_store_count: int = Field(..., description="Number of chunks in vector store")
    embedding_model: str = Field(..., description="Embedding model name")
    llm_model: str = Field(..., description="LLM model name")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
