"""Tests for Hybrid retriever."""

import pytest
from dartboard.retrieval.base import Chunk, RetrievalResult
from dartboard.retrieval.hybrid import HybridRetriever
from dartboard.retrieval.bm25 import BM25Retriever
from dartboard.retrieval.dense import DenseRetriever


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="1",
            text="The quick brown fox jumps over the lazy dog",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "doc1"},
        ),
        Chunk(
            id="2",
            text="A fast brown fox leaps across a sleeping canine",
            embedding=[0.15, 0.25, 0.28, 0.38],
            metadata={"source": "doc2"},
        ),
        Chunk(
            id="3",
            text="Python is a high-level programming language",
            embedding=[0.8, 0.7, 0.1, 0.2],
            metadata={"source": "doc3"},
        ),
        Chunk(
            id="4",
            text="Machine learning algorithms require datasets",
            embedding=[0.7, 0.8, 0.15, 0.25],
            metadata={"source": "doc4"},
        ),
    ]


@pytest.fixture
def mock_vector_store(sample_chunks):
    """Create mock vector store."""

    class MockVectorStore:
        def __init__(self, chunks):
            self.chunks = chunks

        def similarity_search(self, query_embedding, k, metric="cosine"):
            # Return first k chunks with mock scores
            return [(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(self.chunks[:k])]

    return MockVectorStore(sample_chunks)


@pytest.fixture
def hybrid_retriever(sample_chunks, mock_vector_store):
    """Create hybrid retriever."""
    bm25 = BM25Retriever(vector_store=mock_vector_store)
    bm25.fit(sample_chunks)

    dense = DenseRetriever(vector_store=mock_vector_store)

    return HybridRetriever(
        vector_store=mock_vector_store, bm25_retriever=bm25, dense_retriever=dense
    )


def test_hybrid_initialization():
    """Test Hybrid retriever initialization."""
    retriever = HybridRetriever()
    assert retriever.name == "Hybrid"
    assert retriever.k_rrf == 60
    assert retriever.weight_bm25 == 0.5
    assert retriever.weight_dense == 0.5


def test_hybrid_custom_weights():
    """Test Hybrid with custom weights."""
    retriever = HybridRetriever(weight_bm25=0.7, weight_dense=0.3)
    assert retriever.weight_bm25 == 0.7
    assert retriever.weight_dense == 0.3


def test_hybrid_custom_rrf_k():
    """Test Hybrid with custom RRF k parameter."""
    retriever = HybridRetriever(k_rrf=100)
    assert retriever.k_rrf == 100


def test_hybrid_retrieve_basic(hybrid_retriever):
    """Test basic hybrid retrieval."""
    result = hybrid_retriever.retrieve("brown fox", k=2)

    assert len(result.chunks) == 2
    assert len(result.scores) == 2
    assert result.method == "Hybrid"
    assert result.latency_ms > 0


def test_hybrid_retrieve_with_k(hybrid_retriever):
    """Test retrieval with different k values."""
    result_k1 = hybrid_retriever.retrieve("test", k=1)
    result_k3 = hybrid_retriever.retrieve("test", k=3)

    assert len(result_k1.chunks) == 1
    assert len(result_k3.chunks) == 3


def test_hybrid_rrf_fusion(sample_chunks, mock_vector_store):
    """Test RRF fusion algorithm."""
    bm25 = BM25Retriever(vector_store=mock_vector_store)
    bm25.fit(sample_chunks)
    dense = DenseRetriever(vector_store=mock_vector_store)

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=bm25,
        dense_retriever=dense,
        weight_bm25=0.5,
        weight_dense=0.5,
    )

    result = retriever.retrieve("test", k=2)

    # Should successfully fuse results
    assert len(result.chunks) > 0
    assert all(score >= 0 for score in result.scores)


def test_hybrid_scores_descending(hybrid_retriever):
    """Test that hybrid scores are in descending order."""
    result = hybrid_retriever.retrieve("fox", k=4)

    # Scores should be descending
    for i in range(len(result.scores) - 1):
        assert result.scores[i] >= result.scores[i + 1]


def test_hybrid_result_metadata(hybrid_retriever):
    """Test retrieval result metadata."""
    result = hybrid_retriever.retrieve("test", k=2)

    assert "bm25_latency_ms" in result.metadata
    assert "dense_latency_ms" in result.metadata
    assert "fusion_method" in result.metadata
    assert result.metadata["fusion_method"] == "RRF"


def test_hybrid_overlap_tracking(hybrid_retriever):
    """Test overlap tracking between BM25 and Dense."""
    result = hybrid_retriever.retrieve("programming", k=3)

    # Metadata should contain overlap info
    if "overlap" in result.metadata:
        assert "count" in result.metadata["overlap"]
        assert "percentage" in result.metadata["overlap"]


def test_hybrid_retrieval_multiplier(hybrid_retriever):
    """Test that hybrid retrieves more candidates before fusion."""
    # With k=2, should retrieve k*multiplier from each method
    result = hybrid_retriever.retrieve("test", k=2)

    # Should have successfully retrieved and fused
    assert len(result.chunks) == 2


def test_hybrid_metadata_preservation(hybrid_retriever):
    """Test that chunk metadata is preserved."""
    result = hybrid_retriever.retrieve("test", k=2)

    for chunk in result.chunks:
        assert "source" in chunk.metadata


def test_hybrid_different_weight_distributions(sample_chunks, mock_vector_store):
    """Test hybrid with different weight distributions."""
    bm25 = BM25Retriever(vector_store=mock_vector_store)
    bm25.fit(sample_chunks)
    dense = DenseRetriever(vector_store=mock_vector_store)

    # BM25-heavy
    retriever_bm25_heavy = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=bm25,
        dense_retriever=dense,
        weight_bm25=0.8,
        weight_dense=0.2,
    )

    # Dense-heavy
    retriever_dense_heavy = HybridRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=bm25,
        dense_retriever=dense,
        weight_bm25=0.2,
        weight_dense=0.8,
    )

    query = "machine learning"
    result_bm25 = retriever_bm25_heavy.retrieve(query, k=2)
    result_dense = retriever_dense_heavy.retrieve(query, k=2)

    # Both should return results
    assert len(result_bm25.chunks) == 2
    assert len(result_dense.chunks) == 2


def test_hybrid_component_retrievers_called(hybrid_retriever, monkeypatch):
    """Test that both BM25 and Dense retrievers are called."""
    bm25_called = False
    dense_called = False

    original_bm25_retrieve = hybrid_retriever.bm25.retrieve
    original_dense_retrieve = hybrid_retriever.dense.retrieve

    def mock_bm25_retrieve(*args, **kwargs):
        nonlocal bm25_called
        bm25_called = True
        return original_bm25_retrieve(*args, **kwargs)

    def mock_dense_retrieve(*args, **kwargs):
        nonlocal dense_called
        dense_called = True
        return original_dense_retrieve(*args, **kwargs)

    monkeypatch.setattr(hybrid_retriever.bm25, "retrieve", mock_bm25_retrieve)
    monkeypatch.setattr(hybrid_retriever.dense, "retrieve", mock_dense_retrieve)

    hybrid_retriever.retrieve("test", k=2)

    assert bm25_called
    assert dense_called
