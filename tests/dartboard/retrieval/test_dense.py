"""Tests for Dense retriever."""

import pytest
import numpy as np
from dartboard.retrieval.base import Chunk
from dartboard.retrieval.dense import DenseRetriever


@pytest.fixture
def sample_chunks_with_embeddings():
    """Create sample chunks with embeddings."""
    return [
        Chunk(
            id="1",
            text="The quick brown fox",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "doc1"},
        ),
        Chunk(
            id="2",
            text="A fast brown fox",
            embedding=[0.15, 0.25, 0.28, 0.38],
            metadata={"source": "doc2"},
        ),
        Chunk(
            id="3",
            text="Python programming",
            embedding=[0.8, 0.7, 0.1, 0.2],
            metadata={"source": "doc3"},
        ),
        Chunk(
            id="4",
            text="Machine learning",
            embedding=[0.7, 0.8, 0.15, 0.25],
            metadata={"source": "doc4"},
        ),
    ]


@pytest.fixture
def mock_vector_store(sample_chunks_with_embeddings):
    """Create mock vector store."""

    class MockVectorStore:
        def __init__(self, chunks):
            self.chunks = chunks

        def similarity_search(self, query_embedding, k, metric="cosine"):
            # Simple mock: return chunks sorted by random similarity
            results = []
            for chunk in self.chunks[:k]:
                results.append((chunk, 0.9))  # Mock similarity score
            return results

    return MockVectorStore(sample_chunks_with_embeddings)


def test_dense_initialization():
    """Test Dense retriever initialization."""
    retriever = DenseRetriever()
    assert retriever.name == "Dense"
    assert retriever.similarity_metric == "cosine"
    assert retriever.embedding_model is not None


def test_dense_custom_model():
    """Test Dense with custom model name."""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    retriever = DenseRetriever(model_name=model_name)
    assert retriever.model_name == model_name


def test_dense_custom_metric():
    """Test Dense with different similarity metrics."""
    for metric in ["cosine", "dot", "euclidean"]:
        retriever = DenseRetriever(similarity_metric=metric)
        assert retriever.similarity_metric == metric


def test_dense_retrieve_basic(mock_vector_store):
    """Test basic dense retrieval."""
    retriever = DenseRetriever(vector_store=mock_vector_store)
    result = retriever.retrieve("test query", k=2)

    assert len(result.chunks) == 2
    assert len(result.scores) == 2
    assert result.method == "Dense"
    assert result.latency_ms > 0


def test_dense_retrieve_with_k(mock_vector_store):
    """Test retrieval with different k values."""
    retriever = DenseRetriever(vector_store=mock_vector_store)

    result_k1 = retriever.retrieve("test", k=1)
    result_k3 = retriever.retrieve("test", k=3)

    assert len(result_k1.chunks) == 1
    assert len(result_k3.chunks) == 3


def test_dense_query_encoding(mock_vector_store):
    """Test that queries are properly encoded."""
    retriever = DenseRetriever(vector_store=mock_vector_store)

    # Should not raise error
    result = retriever.retrieve("This is a test query", k=2)
    assert len(result.chunks) == 2


def test_dense_metadata_preservation(mock_vector_store):
    """Test that chunk metadata is preserved."""
    retriever = DenseRetriever(vector_store=mock_vector_store)
    result = retriever.retrieve("test", k=2)

    for chunk in result.chunks:
        assert "source" in chunk.metadata


def test_dense_result_metadata(mock_vector_store):
    """Test retrieval result metadata."""
    retriever = DenseRetriever(vector_store=mock_vector_store)
    result = retriever.retrieve("test", k=2)

    assert "embedding_model" in result.metadata
    assert "similarity_metric" in result.metadata
    assert result.metadata["similarity_metric"] == "cosine"


def test_dense_batch_encoding():
    """Test batch encoding of multiple queries."""
    retriever = DenseRetriever()

    queries = ["query 1", "query 2", "query 3"]
    embeddings = retriever.embedding_model.encode(queries)

    assert len(embeddings) == 3
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)


def test_dense_embedding_dimension():
    """Test that embeddings have correct dimensions."""
    retriever = DenseRetriever()
    query = "test query"

    embedding = retriever.embedding_model.encode(query)

    # Default model produces 384-dimensional embeddings
    assert len(embedding) == 384


def test_dense_similarity_scores_normalized(mock_vector_store):
    """Test that similarity scores are in valid range."""
    retriever = DenseRetriever(vector_store=mock_vector_store)
    result = retriever.retrieve("test", k=2)

    # Scores should be between 0 and 1 for cosine similarity
    for score in result.scores:
        assert 0 <= score <= 1


def test_dense_empty_query(mock_vector_store):
    """Test behavior with empty query."""
    retriever = DenseRetriever(vector_store=mock_vector_store)

    # Should handle gracefully
    result = retriever.retrieve("", k=2)
    assert len(result.chunks) >= 0
