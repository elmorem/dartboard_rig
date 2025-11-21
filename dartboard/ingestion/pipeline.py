"""
End-to-end ingestion pipeline for Dartboard RAG.

Orchestrates the full document processing workflow:
1. Load documents from source
2. Chunk documents into passages
3. Generate embeddings for chunks
4. Store chunks in vector database
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from dartboard.ingestion.loaders import DocumentLoader, Document
from dartboard.ingestion.chunking import RecursiveChunker
from dartboard.embeddings import EmbeddingModel
from dartboard.storage.vector_store import VectorStore
from dartboard.datasets.models import Chunk as StorageChunk

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingestion operation."""

    documents_processed: int
    chunks_created: int
    chunks_stored: int
    status: str
    errors: List[str]
    metadata: Dict[str, Any]


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Orchestrates: Load → Chunk → Embed → Store
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: RecursiveChunker,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        batch_size: int = 32,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            loader: Document loader (PDF, Markdown, Code)
            chunker: Chunking strategy
            embedding_model: Embedding model
            vector_store: Vector database
            batch_size: Batch size for embedding generation
        """
        self.loader = loader
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.batch_size = batch_size

    def ingest(self, source: str, track_progress: bool = False) -> IngestionResult:
        """
        Ingest document from source.

        Args:
            source: Document source (file path, directory, URL)
            track_progress: Whether to log progress

        Returns:
            IngestionResult with statistics and status
        """
        errors = []
        metadata = {}

        try:
            # Step 1: Load documents
            if track_progress:
                logger.info(f"Loading documents from {source}")

            documents = self._load_documents(source)

            if not documents or len(documents) == 0:
                return IngestionResult(
                    documents_processed=0,
                    chunks_created=0,
                    chunks_stored=0,
                    status="failed",
                    errors=["No documents loaded"],
                    metadata=metadata,
                )

            if track_progress:
                logger.info(f"Loaded {len(documents)} document(s)")

            # Step 2: Chunk documents
            if track_progress:
                logger.info("Chunking documents...")

            all_chunks = self._chunk_documents(documents, track_progress)

            if track_progress:
                logger.info(f"Created {len(all_chunks)} chunk(s)")

            # Step 3: Generate embeddings
            if track_progress:
                logger.info("Generating embeddings...")

            chunks_with_embeddings = self._embed_chunks(all_chunks, track_progress)

            if track_progress:
                logger.info(f"Generated {len(chunks_with_embeddings)} embeddings")

            # Step 4: Store in vector database
            if track_progress:
                logger.info("Storing chunks in vector database...")

            chunks_stored = self._store_chunks(chunks_with_embeddings)

            if track_progress:
                logger.info(f"Stored {chunks_stored} chunk(s)")

            # Build result
            return IngestionResult(
                documents_processed=len(documents),
                chunks_created=len(all_chunks),
                chunks_stored=chunks_stored,
                status="success",
                errors=errors,
                metadata={
                    "source": source,
                    "avg_chunk_size": np.mean(
                        [len(c.text) for c in all_chunks if hasattr(c, "text")]
                    ),
                    **metadata,
                },
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            errors.append(str(e))
            return IngestionResult(
                documents_processed=0,
                chunks_created=0,
                chunks_stored=0,
                status="failed",
                errors=errors,
                metadata=metadata,
            )

    def ingest_batch(
        self, sources: List[str], track_progress: bool = False
    ) -> List[IngestionResult]:
        """
        Ingest multiple documents.

        Args:
            sources: List of document sources
            track_progress: Whether to log progress

        Returns:
            List of IngestionResult for each source
        """
        results = []

        for i, source in enumerate(sources):
            if track_progress:
                logger.info(f"Processing {i+1}/{len(sources)}: {source}")

            result = self.ingest(source, track_progress=False)
            results.append(result)

            if track_progress and result.status == "failed":
                logger.warning(f"Failed to process {source}: {result.errors}")

        if track_progress:
            total_docs = sum(r.documents_processed for r in results)
            total_chunks = sum(r.chunks_created for r in results)
            logger.info(
                f"Batch complete: {total_docs} documents, {total_chunks} chunks"
            )

        return results

    def _load_documents(self, source: str) -> List[Document]:
        """Load documents from source."""
        try:
            return self.loader.load(source)
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def _chunk_documents(
        self, documents: List[Document], track_progress: bool = False
    ) -> List[Any]:
        """Chunk documents into passages."""
        all_chunks = []

        for i, doc in enumerate(documents):
            try:
                chunks = self.chunker.chunk(doc)
                all_chunks.extend(chunks)

                if track_progress and (i + 1) % 10 == 0:
                    logger.info(f"Chunked {i+1}/{len(documents)} documents")

            except Exception as e:
                logger.error(f"Failed to chunk document {doc.source}: {e}")
                # Continue with other documents
                continue

        return all_chunks

    def _embed_chunks(
        self, chunks: List[Any], track_progress: bool = False
    ) -> List[StorageChunk]:
        """Generate embeddings for chunks."""
        storage_chunks = []

        # Process in batches for efficiency
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]

            try:
                # Extract texts
                texts = [chunk.text for chunk in batch]

                # Generate embeddings
                embeddings = self.embedding_model.encode(texts)

                # Ensure 2D array
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)

                # Create StorageChunk objects
                for j, chunk in enumerate(batch):
                    storage_chunk = StorageChunk(
                        id=f"{chunk.metadata.get('source', 'unknown')}_{chunk.chunk_index}",
                        text=chunk.text,
                        embedding=embeddings[j],
                        metadata=chunk.metadata,
                    )
                    storage_chunks.append(storage_chunk)

                if track_progress and (i + self.batch_size) % 100 == 0:
                    logger.info(
                        f"Embedded {min(i + self.batch_size, len(chunks))}/{len(chunks)} chunks"
                    )

            except Exception as e:
                logger.error(f"Failed to embed batch at index {i}: {e}")
                # Continue with other batches
                continue

        return storage_chunks

    def _store_chunks(self, chunks: List[StorageChunk]) -> int:
        """Store chunks in vector database."""
        try:
            self.vector_store.add(chunks)
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise


class BatchIngestionPipeline(IngestionPipeline):
    """
    Pipeline optimized for batch ingestion.

    Processes multiple documents with:
    - Parallel loading
    - Batch embedding generation
    - Progress tracking
    - Error recovery
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: RecursiveChunker,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        batch_size: int = 32,
        max_retries: int = 3,
    ):
        """
        Initialize batch ingestion pipeline.

        Args:
            loader: Document loader
            chunker: Chunking strategy
            embedding_model: Embedding model
            vector_store: Vector database
            batch_size: Batch size for embedding generation
            max_retries: Maximum retries for failed operations
        """
        super().__init__(loader, chunker, embedding_model, vector_store, batch_size)
        self.max_retries = max_retries

    def ingest_with_retry(
        self, source: str, track_progress: bool = False
    ) -> IngestionResult:
        """
        Ingest document with retry logic.

        Args:
            source: Document source
            track_progress: Whether to log progress

        Returns:
            IngestionResult
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if track_progress and attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")

                result = self.ingest(source, track_progress)

                if result.status == "success":
                    return result

                last_error = result.errors

            except Exception as e:
                last_error = [str(e)]
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

        # All retries failed
        return IngestionResult(
            documents_processed=0,
            chunks_created=0,
            chunks_stored=0,
            status="failed",
            errors=last_error or ["Max retries exceeded"],
            metadata={"source": source, "attempts": self.max_retries},
        )


def create_pipeline(
    loader: DocumentLoader,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    chunk_size: int = 512,
    overlap: int = 50,
    batch_size: int = 32,
) -> IngestionPipeline:
    """
    Factory function to create a standard ingestion pipeline.

    Args:
        loader: Document loader
        embedding_model: Embedding model
        vector_store: Vector database
        chunk_size: Maximum tokens per chunk
        overlap: Token overlap between chunks
        batch_size: Batch size for embeddings

    Returns:
        Configured IngestionPipeline
    """
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        respect_code_blocks=True,
    )

    return IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
        batch_size=batch_size,
    )
