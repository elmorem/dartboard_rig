"""
Tests for FastAPI routes.

Tests:
- /query endpoint
- /ingest endpoint
- /health endpoint
- Error handling
- Request validation
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from dartboard.api.routes import app
from dartboard.datasets.models import Chunk
from dartboard.generation.generator import GenerationResult
from dartboard.ingestion.pipeline import IngestionResult


# Note: Test client created in each test to allow dependency overrides


# Sample test data
SAMPLE_CHUNKS = [
    Chunk(
        id="chunk1",
        text="Machine learning is a subset of artificial intelligence.",
        embedding=np.random.rand(384).astype(np.float32),
        metadata={"source": "ml_intro.pdf", "chunk_index": 0},
    ),
    Chunk(
        id="chunk2",
        text="Deep learning uses neural networks.",
        embedding=np.random.rand(384).astype(np.float32),
        metadata={"source": "ml_intro.pdf", "chunk_index": 1},
    ),
]


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self):
        """Test root endpoint returns API info."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "Dartboard" in data["name"]
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self):
        """Test health check endpoint."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.count.return_value = 100

        mock_config_func = lambda: {
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gpt-3.5-turbo",
        }

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_vector_store] = lambda: mock_store
        app.dependency_overrides[routes.get_config] = mock_config_func

        client = TestClient(app)

        # Make request
        response = client.get("/health")

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store_count"] == 100
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        assert data["llm_model"] == "gpt-3.5-turbo"

    def test_health_check_empty_store(self):
        """Test health check with empty vector store."""
        mock_store = MagicMock()
        mock_store.count.return_value = 0

        mock_config_func = lambda: {
            "embedding_model": "test-model",
            "llm_model": "test-llm",
        }

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_vector_store] = lambda: mock_store
        app.dependency_overrides[routes.get_config] = mock_config_func

        client = TestClient(app)
        response = client.get("/health")

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200
        data = response.json()
        assert data["vector_store_count"] == 0


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_success(self):
        """Test successful query."""
        # Setup retriever mock
        mock_retriever = MagicMock()
        mock_retrieval_result = MagicMock()
        mock_retrieval_result.chunks = SAMPLE_CHUNKS
        mock_retriever.retrieve.return_value = mock_retrieval_result

        # Setup generator mock
        mock_generator = MagicMock()
        mock_generation_result = GenerationResult(
            answer="Machine learning is a subset of AI [Source 1].",
            sources=[
                {
                    "number": 1,
                    "text": SAMPLE_CHUNKS[0].text,
                    "metadata": SAMPLE_CHUNKS[0].metadata,
                }
            ],
            num_sources_cited=1,
            total_sources_available=2,
            model="gpt-3.5-turbo",
            metadata={},
        )
        mock_generator.generate.return_value = mock_generation_result

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_hybrid_retriever] = lambda: mock_retriever
        app.dependency_overrides[routes.get_rag_generator] = lambda: mock_generator

        client = TestClient(app)

        # Make request
        response = client.post(
            "/query",
            json={
                "query": "What is machine learning?",
                "top_k": 5,
                "sigma": 1.0,
                "temperature": 0.7,
            },
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "Machine learning" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["num_sources_cited"] == 1
        assert "retrieval_time_ms" in data
        assert "generation_time_ms" in data
        assert "total_time_ms" in data

    def test_query_no_results(self):
        """Test query with no retrieval results."""
        mock_retriever = MagicMock()
        mock_retrieval_result = MagicMock()
        mock_retrieval_result.chunks = []
        mock_retriever.retrieve.return_value = mock_retrieval_result

        mock_generator = MagicMock()  # Won't be called but need to override

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_hybrid_retriever] = lambda: mock_retriever
        app.dependency_overrides[routes.get_rag_generator] = lambda: mock_generator

        client = TestClient(app)
        response = client.post("/query", json={"query": "What is XYZ?", "top_k": 5})

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 404
        detail = response.json()["detail"].lower()
        assert "not found" in detail or "no relevant documents" in detail

    def test_query_invalid_request(self):
        """Test query with invalid request data."""
        # Override dependencies even though we're testing validation
        # (Pydantic validation happens before dependencies are called)
        mock_retriever = MagicMock()
        mock_generator = MagicMock()

        from dartboard.api import routes

        app.dependency_overrides[routes.get_hybrid_retriever] = lambda: mock_retriever
        app.dependency_overrides[routes.get_rag_generator] = lambda: mock_generator

        client = TestClient(app)

        # Empty query
        response = client.post("/query", json={"query": "", "top_k": 5})
        assert response.status_code == 422

        # Invalid top_k
        response = client.post("/query", json={"query": "test", "top_k": 0})
        assert response.status_code == 422

        # Invalid sigma
        response = client.post(
            "/query", json={"query": "test", "top_k": 5, "sigma": -1}
        )
        assert response.status_code == 422

        # Clear overrides
        app.dependency_overrides = {}

    def test_query_with_custom_params(self):
        """Test query with custom parameters."""
        mock_retriever = MagicMock()
        mock_retrieval_result = MagicMock()
        mock_retrieval_result.chunks = SAMPLE_CHUNKS
        mock_retriever.retrieve.return_value = mock_retrieval_result

        mock_generator = MagicMock()
        mock_generation_result = GenerationResult(
            answer="Test answer",
            sources=[],
            num_sources_cited=0,
            total_sources_available=2,
            model="gpt-4",
            metadata={},
        )
        mock_generator.generate.return_value = mock_generation_result

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_hybrid_retriever] = lambda: mock_retriever
        app.dependency_overrides[routes.get_rag_generator] = lambda: mock_generator

        client = TestClient(app)
        response = client.post(
            "/query",
            json={
                "query": "test",
                "top_k": 10,
                "sigma": 2.0,
                "temperature": 0.5,
            },
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200

        # Verify parameters passed to retriever
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["k"] == 10
        assert call_kwargs["sigma"] == 2.0

        # Verify parameters passed to generator
        mock_generator.generate.assert_called_once()
        gen_call_kwargs = mock_generator.generate.call_args.kwargs
        assert gen_call_kwargs["temperature"] == 0.5


class TestIngestEndpoint:
    """Tests for /ingest endpoint."""

    def test_ingest_markdown_file(self):
        """Test ingesting markdown file."""
        # Setup pipeline mock
        mock_pipeline = MagicMock()
        mock_result = IngestionResult(
            documents_processed=1,
            chunks_created=5,
            chunks_stored=5,
            status="success",
            errors=[],
            metadata={"avg_chunk_size": 250},
        )
        mock_pipeline.ingest.return_value = mock_result
        mock_pipeline.chunker = MagicMock()

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_ingestion_pipeline] = lambda: mock_pipeline

        client = TestClient(app)

        # Create test file
        content = b"# Test Document\n\nThis is a test markdown file."

        # Make request
        response = client.post(
            "/ingest",
            files={"file": ("test.md", content, "text/markdown")},
            data={"chunk_size": 512, "overlap": 50, "track_progress": False},
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 5
        assert data["chunks_stored"] == 5
        assert "processing_time_ms" in data
        assert data["metadata"]["filename"] == "test.md"

    def test_ingest_text_file(self):
        """Test ingesting text file."""
        mock_pipeline = MagicMock()
        mock_result = IngestionResult(
            documents_processed=1,
            chunks_created=3,
            chunks_stored=3,
            status="success",
            errors=[],
            metadata={},
        )
        mock_pipeline.ingest.return_value = mock_result
        mock_pipeline.chunker = MagicMock()

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_ingestion_pipeline] = lambda: mock_pipeline

        client = TestClient(app)
        content = b"Plain text content for testing."

        response = client.post(
            "/ingest",
            files={"file": ("test.txt", content, "text/plain")},
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_ingest_unsupported_file_type(self):
        """Test ingesting unsupported file type."""
        # Override dependencies even though validation happens before they're called
        mock_pipeline = MagicMock()
        from dartboard.api import routes

        app.dependency_overrides[routes.get_ingestion_pipeline] = lambda: mock_pipeline

        client = TestClient(app)
        content = b"Binary content"

        response = client.post(
            "/ingest",
            files={"file": ("test.exe", content, "application/octet-stream")},
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_ingest_with_custom_params(self):
        """Test ingestion with custom chunk parameters."""
        mock_pipeline = MagicMock()
        mock_result = IngestionResult(
            documents_processed=1,
            chunks_created=10,
            chunks_stored=10,
            status="success",
            errors=[],
            metadata={},
        )
        mock_pipeline.ingest.return_value = mock_result
        mock_pipeline.chunker = MagicMock()

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_ingestion_pipeline] = lambda: mock_pipeline

        client = TestClient(app)
        content = b"# Test\n\nContent"

        response = client.post(
            "/ingest",
            files={"file": ("test.md", content)},
            data={"chunk_size": 256, "overlap": 25, "track_progress": True},
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200

        # Verify chunker was configured
        assert mock_pipeline.chunker.chunk_size == 256
        assert mock_pipeline.chunker.overlap == 25

    def test_ingest_with_errors(self):
        """Test ingestion with errors."""
        mock_pipeline = MagicMock()
        mock_result = IngestionResult(
            documents_processed=0,
            chunks_created=0,
            chunks_stored=0,
            status="failed",
            errors=["Failed to load document"],
            metadata={},
        )
        mock_pipeline.ingest.return_value = mock_result
        mock_pipeline.chunker = MagicMock()

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_ingestion_pipeline] = lambda: mock_pipeline

        client = TestClient(app)
        content = b"# Test"

        response = client.post(
            "/ingest",
            files={"file": ("test.md", content)},
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 200  # Returns 200 but status=failed
        data = response.json()
        assert data["status"] == "failed"
        assert len(data["errors"]) > 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_query_internal_error(self):
        """Test query with internal error."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("Internal error")

        mock_generator = MagicMock()  # Won't be called but need to override

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_hybrid_retriever] = lambda: mock_retriever
        app.dependency_overrides[routes.get_rag_generator] = lambda: mock_generator

        client = TestClient(app)
        response = client.post("/query", json={"query": "test", "top_k": 5})

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()

    def test_ingest_internal_error(self):
        """Test ingest with internal error."""
        mock_pipeline = MagicMock()
        mock_pipeline.ingest.side_effect = Exception("Processing error")
        mock_pipeline.chunker = MagicMock()

        # Override dependencies
        from dartboard.api import routes

        app.dependency_overrides[routes.get_ingestion_pipeline] = lambda: mock_pipeline

        client = TestClient(app)
        content = b"# Test"

        response = client.post(
            "/ingest",
            files={"file": ("test.md", content)},
        )

        # Clear overrides
        app.dependency_overrides = {}

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


class TestRequestValidation:
    """Tests for Pydantic request validation."""

    def test_query_request_validation(self):
        """Test query request validation."""
        # Override dependencies
        mock_retriever = MagicMock()
        mock_generator = MagicMock()
        from dartboard.api import routes

        app.dependency_overrides[routes.get_hybrid_retriever] = lambda: mock_retriever
        app.dependency_overrides[routes.get_rag_generator] = lambda: mock_generator

        client = TestClient(app)

        # Missing required field
        response = client.post("/query", json={"top_k": 5})
        assert response.status_code == 422

        # Invalid type
        response = client.post("/query", json={"query": 123, "top_k": 5})
        assert response.status_code == 422

        # Out of range
        response = client.post("/query", json={"query": "test", "top_k": 100})
        assert response.status_code == 422

        # Clear overrides
        app.dependency_overrides = {}

    def test_ingest_form_validation(self):
        """Test ingest form validation."""
        # Override dependencies with a properly configured mock
        mock_pipeline = MagicMock()
        mock_result = IngestionResult(
            documents_processed=1,
            chunks_created=1,
            chunks_stored=1,
            status="success",
            errors=[],
            metadata={},
        )
        mock_pipeline.ingest.return_value = mock_result
        mock_pipeline.chunker = MagicMock()

        from dartboard.api import routes

        app.dependency_overrides[routes.get_ingestion_pipeline] = lambda: mock_pipeline

        client = TestClient(app)
        content = b"# Test"

        # Invalid chunk_size (too small - minimum is 100)
        response = client.post(
            "/ingest",
            files={"file": ("test.md", content)},
            data={"chunk_size": 50},  # Too small
        )
        assert response.status_code == 422

        # Clear overrides
        app.dependency_overrides = {}


def test_api_documentation():
    """Test that API docs are available."""
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200


if __name__ == "__main__":
    print("Running API tests...")
    pytest.main([__file__, "-v"])
