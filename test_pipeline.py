"""
Tests for end-to-end ingestion pipeline.

Tests:
- Basic ingestion flow (Load → Chunk → Embed → Store)
- Batch processing
- Progress tracking
- Error handling and retry logic
- Integration with different loaders and chunkers
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from dartboard.ingestion.pipeline import (
    IngestionPipeline,
    BatchIngestionPipeline,
    IngestionResult,
    create_pipeline,
)
from dartboard.ingestion.loaders import Document, MarkdownLoader
from dartboard.ingestion.chunking import RecursiveChunker, Document as ChunkDoc
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore
from dartboard.datasets.models import Chunk as StorageChunk


# Sample content for testing
SAMPLE_MARKDOWN = """
# Machine Learning

Machine learning is a subset of AI. It learns from data.

## Deep Learning

Deep learning uses neural networks. Networks have multiple layers.

## Applications

ML powers many modern applications. Examples include search and recommendations.
""".strip()


class MockLoader:
    """Mock document loader for testing."""

    def __init__(self, documents=None):
        self.documents = documents or []

    def load(self, source: str):
        if not self.documents:
            # Return default test document
            return [
                Document(
                    content=SAMPLE_MARKDOWN,
                    metadata={"source": source, "title": "Test Doc"},
                    source=source,
                )
            ]
        return self.documents


class MockChunker:
    """Mock chunker for testing."""

    def __init__(self, num_chunks=3):
        self.num_chunks = num_chunks

    def chunk(self, document):
        # Split into equal parts
        text = document.content
        chunk_size = len(text) // self.num_chunks
        chunks = []

        for i in range(self.num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_chunks - 1 else len(text)
            chunk_text = text[start:end]

            chunk = Mock()
            chunk.text = chunk_text
            chunk.metadata = {**document.metadata, "chunk_index": i}
            chunk.chunk_index = i

            chunks.append(chunk)

        return chunks


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts):
        if isinstance(texts, str):
            return np.random.rand(self.dim).astype(np.float32)
        return np.random.rand(len(texts), self.dim).astype(np.float32)


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.chunks = []

    def add(self, chunks):
        self.chunks.extend(chunks)

    def count(self):
        return len(self.chunks)

    def search(self, query_embedding, k=5, filters=None):
        return self.chunks[:k]


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    def test_basic_ingestion(self):
        """Test basic end-to-end ingestion."""
        # Setup pipeline
        loader = MockLoader()
        chunker = MockChunker(num_chunks=3)
        embedding_model = MockEmbeddingModel(dim=384)
        vector_store = MockVectorStore(embedding_dim=384)

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Ingest document
        result = pipeline.ingest("test.md")

        # Verify result
        assert isinstance(result, IngestionResult)
        assert result.status == "success"
        assert result.documents_processed == 1
        assert result.chunks_created == 3
        assert result.chunks_stored == 3
        assert len(result.errors) == 0

    def test_ingestion_result_metadata(self):
        """Test that ingestion result contains metadata."""
        loader = MockLoader()
        chunker = MockChunker(num_chunks=5)
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        result = pipeline.ingest("test.md")

        # Check metadata
        assert "source" in result.metadata
        assert result.metadata["source"] == "test.md"
        assert "avg_chunk_size" in result.metadata

    def test_batch_processing(self):
        """Test batch size configuration."""
        loader = MockLoader()
        chunker = MockChunker(num_chunks=10)
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        # Test different batch sizes
        for batch_size in [1, 5, 32]:
            pipeline = IngestionPipeline(
                loader=loader,
                chunker=chunker,
                embedding_model=embedding_model,
                vector_store=MockVectorStore(),  # Fresh store
                batch_size=batch_size,
            )

            result = pipeline.ingest("test.md")
            assert result.chunks_stored == 10

    def test_progress_tracking(self, caplog):
        """Test progress tracking logs."""
        loader = MockLoader()
        chunker = MockChunker(num_chunks=3)
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Ingest with progress tracking
        with caplog.at_level("INFO"):
            result = pipeline.ingest("test.md", track_progress=True)

        # Should have logged progress
        assert result.status == "success"
        # Check that at least some logging happened
        assert len(caplog.records) > 0

    def test_empty_document(self):
        """Test handling of empty document."""
        # Create loader with empty document
        empty_doc = Document(content="", metadata={}, source="empty.txt")
        loader = MockLoader(documents=[empty_doc])
        chunker = MockChunker(num_chunks=1)
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        result = pipeline.ingest("empty.txt")

        # Should handle gracefully
        assert result.documents_processed == 1

    def test_no_documents_loaded(self):
        """Test handling when no documents are loaded."""

        class EmptyLoader:
            def load(self, source):
                return []  # Return truly empty list

        loader = EmptyLoader()
        chunker = MockChunker()
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        result = pipeline.ingest("nonexistent.txt")

        assert result.status == "failed"
        assert result.documents_processed == 0
        assert len(result.errors) > 0

    def test_error_handling(self):
        """Test error handling in pipeline."""

        class FailingLoader:
            def load(self, source):
                raise Exception("Load failed")

        loader = FailingLoader()
        chunker = MockChunker()
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        result = pipeline.ingest("test.md")

        assert result.status == "failed"
        assert len(result.errors) > 0
        assert "Load failed" in result.errors[0]

    def test_chunking_error_recovery(self):
        """Test that pipeline continues after chunking error."""

        class PartiallyFailingChunker:
            def __init__(self):
                self.call_count = 0

            def chunk(self, document):
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("Chunking failed")
                # Succeed for other documents
                return [Mock(text="chunk", metadata={}, chunk_index=0)]

        # Create multiple documents
        docs = [
            Document(content="doc1", metadata={}, source="doc1.txt"),
            Document(content="doc2", metadata={}, source="doc2.txt"),
        ]
        loader = MockLoader(documents=docs)
        chunker = PartiallyFailingChunker()
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        result = pipeline.ingest("test")

        # Should process second document despite first failing
        assert result.status == "success"
        assert result.chunks_created >= 1


class TestBatchIngestionPipeline:
    """Tests for BatchIngestionPipeline."""

    def test_ingest_multiple_sources(self):
        """Test ingesting multiple sources."""
        loader = MockLoader()
        chunker = MockChunker(num_chunks=2)
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        sources = ["doc1.md", "doc2.md", "doc3.md"]
        results = pipeline.ingest_batch(sources)

        assert len(results) == 3
        assert all(r.status == "success" for r in results)
        assert sum(r.documents_processed for r in results) == 3

    def test_retry_logic(self):
        """Test retry logic in BatchIngestionPipeline."""

        class FlakyLoader:
            def __init__(self):
                self.attempt = 0

            def load(self, source):
                self.attempt += 1
                if self.attempt < 2:
                    raise Exception("Temporary failure")
                return [Document(content="success", metadata={}, source=source)]

        loader = FlakyLoader()
        chunker = MockChunker()
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = BatchIngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
            max_retries=3,
        )

        result = pipeline.ingest_with_retry("test.md")

        # Should succeed on retry
        assert result.status == "success"

    def test_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""

        class AlwaysFailingLoader:
            def load(self, source):
                raise Exception("Permanent failure")

        loader = AlwaysFailingLoader()
        chunker = MockChunker()
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = BatchIngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
            max_retries=2,
        )

        result = pipeline.ingest_with_retry("test.md")

        assert result.status == "failed"
        assert len(result.errors) > 0


class TestPipelineFactory:
    """Tests for pipeline factory function."""

    def test_create_pipeline(self):
        """Test pipeline creation with factory."""
        loader = MockLoader()
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = create_pipeline(
            loader=loader,
            embedding_model=embedding_model,
            vector_store=vector_store,
            chunk_size=256,
            overlap=25,
            batch_size=16,
        )

        assert isinstance(pipeline, IngestionPipeline)
        assert pipeline.batch_size == 16
        assert isinstance(pipeline.chunker, RecursiveChunker)
        assert pipeline.chunker.chunk_size == 256
        assert pipeline.chunker.overlap == 25


class TestIntegrationWithRealComponents:
    """Integration tests with real components (if available)."""

    def test_integration_with_markdown_loader(self, tmp_path):
        """Test integration with real MarkdownLoader."""
        # Create test markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text(SAMPLE_MARKDOWN)

        # Setup pipeline with real components
        loader = MarkdownLoader()
        chunker = RecursiveChunker(chunk_size=100, overlap=20)
        embedding_model = MockEmbeddingModel()  # Keep mock to avoid model download
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Ingest
        result = pipeline.ingest(str(test_file))

        # Verify
        assert result.status == "success"
        assert result.documents_processed == 1
        assert result.chunks_created > 0
        assert result.chunks_stored > 0

    def test_end_to_end_with_real_chunker(self):
        """Test end-to-end with real RecursiveChunker."""
        loader = MockLoader()
        chunker = RecursiveChunker(chunk_size=50, overlap=10)
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        result = pipeline.ingest("test.md")

        # Verify chunking worked
        assert result.status == "success"
        assert result.chunks_created > 1  # Should split into multiple chunks


def test_pipeline_full_workflow():
    """Test complete ingestion workflow."""
    # Create test document
    loader = MockLoader()
    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    embedding_model = MockEmbeddingModel()
    vector_store = MockVectorStore()

    # Create pipeline
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
        batch_size=4,
    )

    # Ingest
    result = pipeline.ingest("test.md", track_progress=True)

    # Verify all steps completed
    assert result.status == "success"
    assert result.documents_processed == 1
    assert result.chunks_created > 0
    assert result.chunks_stored == result.chunks_created
    assert vector_store.count() == result.chunks_stored


if __name__ == "__main__":
    print("Running pipeline tests...")
    pytest.main([__file__, "-v"])
