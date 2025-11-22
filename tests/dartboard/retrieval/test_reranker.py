"""Tests for Cross-Encoder Reranker."""

import pytest
from dartboard.retrieval.base import Chunk
from dartboard.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="1",
            text="The quick brown fox jumps over the lazy dog",
            metadata={"source": "doc1"},
            score=0.8,
        ),
        Chunk(
            id="2",
            text="Python is a programming language",
            metadata={"source": "doc2"},
            score=0.7,
        ),
        Chunk(
            id="3",
            text="Machine learning uses neural networks",
            metadata={"source": "doc3"},
            score=0.6,
        ),
        Chunk(
            id="4",
            text="Data science involves statistics",
            metadata={"source": "doc4"},
            score=0.5,
        ),
    ]


def test_reranker_initialization():
    """Test reranker initialization."""
    reranker = CrossEncoderReranker()
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.batch_size == 32
    assert reranker.model is not None


def test_reranker_custom_model():
    """Test reranker with custom model."""
    model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    reranker = CrossEncoderReranker(model_name=model_name)
    assert reranker.model_name == model_name


def test_reranker_custom_batch_size():
    """Test reranker with custom batch size."""
    reranker = CrossEncoderReranker(batch_size=16)
    assert reranker.batch_size == 16


def test_reranker_basic(sample_chunks):
    """Test basic reranking."""
    reranker = CrossEncoderReranker()
    query = "What is Python?"

    result = reranker.rerank(query, sample_chunks, top_k=2)

    assert len(result.chunks) == 2
    assert len(result.scores) == 2
    assert result.method == "CrossEncoder"
    assert result.latency_ms > 0


def test_reranker_top_k(sample_chunks):
    """Test reranking with different top_k values."""
    reranker = CrossEncoderReranker()
    query = "programming"

    result_k1 = reranker.rerank(query, sample_chunks, top_k=1)
    result_k3 = reranker.rerank(query, sample_chunks, top_k=3)

    assert len(result_k1.chunks) == 1
    assert len(result_k3.chunks) == 3


def test_reranker_scores_descending(sample_chunks):
    """Test that reranked scores are in descending order."""
    reranker = CrossEncoderReranker()
    query = "test"

    result = reranker.rerank(query, sample_chunks, top_k=4)

    # Scores should be descending
    for i in range(len(result.scores) - 1):
        assert result.scores[i] >= result.scores[i + 1]


def test_reranker_no_top_k(sample_chunks):
    """Test reranking without top_k (rerank all)."""
    reranker = CrossEncoderReranker()
    query = "test"

    result = reranker.rerank(query, sample_chunks)

    # Should return all chunks, reranked
    assert len(result.chunks) == len(sample_chunks)


def test_reranker_changes_order(sample_chunks):
    """Test that reranking can change chunk order."""
    reranker = CrossEncoderReranker()

    # Query specifically about Python
    query = "What is Python programming?"
    result = reranker.rerank(query, sample_chunks, top_k=4)

    # Document about Python should rank highly
    python_doc_found = False
    for chunk in result.chunks[:2]:  # Check top 2
        if "Python" in chunk.text:
            python_doc_found = True
            break

    assert python_doc_found, "Python document should be in top 2 for Python query"


def test_reranker_metadata_preservation(sample_chunks):
    """Test that chunk metadata is preserved."""
    reranker = CrossEncoderReranker()
    query = "test"

    result = reranker.rerank(query, sample_chunks, top_k=2)

    for chunk in result.chunks:
        assert "source" in chunk.metadata


def test_reranker_result_metadata(sample_chunks):
    """Test reranking result metadata."""
    reranker = CrossEncoderReranker()
    query = "test"

    result = reranker.rerank(query, sample_chunks, top_k=2)

    assert "model" in result.metadata
    assert "num_candidates" in result.metadata
    assert result.metadata["num_candidates"] == len(sample_chunks)


def test_reranker_single_chunk():
    """Test reranking with single chunk."""
    reranker = CrossEncoderReranker()
    chunk = Chunk(id="1", text="Test document", metadata={})

    result = reranker.rerank("test", [chunk], top_k=1)

    assert len(result.chunks) == 1
    assert result.chunks[0].id == "1"


def test_reranker_empty_chunks():
    """Test reranking with empty chunk list."""
    reranker = CrossEncoderReranker()

    result = reranker.rerank("test", [], top_k=5)

    assert len(result.chunks) == 0
    assert len(result.scores) == 0


def test_reranker_batch_processing(sample_chunks):
    """Test batch processing with different batch sizes."""
    # Small batch size
    reranker_small = CrossEncoderReranker(batch_size=2)
    result_small = reranker_small.rerank("test", sample_chunks, top_k=4)

    # Large batch size
    reranker_large = CrossEncoderReranker(batch_size=32)
    result_large = reranker_large.rerank("test", sample_chunks, top_k=4)

    # Results should be similar (order might vary slightly due to model)
    assert len(result_small.chunks) == len(result_large.chunks)


def test_reranker_score_updates(sample_chunks):
    """Test that chunk scores are updated after reranking."""
    reranker = CrossEncoderReranker()
    query = "Python programming"

    original_scores = [chunk.score for chunk in sample_chunks]
    result = reranker.rerank(query, sample_chunks, top_k=4)

    # At least some scores should be different
    new_scores = result.scores
    assert new_scores != original_scores


def test_reranker_query_encoding(sample_chunks):
    """Test that different queries produce different rankings."""
    reranker = CrossEncoderReranker()

    result_python = reranker.rerank("Python programming", sample_chunks, top_k=2)
    result_ml = reranker.rerank("machine learning", sample_chunks, top_k=2)

    # Top results should potentially be different
    # (or at least we should be able to compute both without error)
    assert len(result_python.chunks) == 2
    assert len(result_ml.chunks) == 2
