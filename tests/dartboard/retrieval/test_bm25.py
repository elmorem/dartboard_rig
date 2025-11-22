"""Tests for BM25 retriever."""

import pytest
from dartboard.retrieval.base import Chunk
from dartboard.retrieval.bm25 import BM25Retriever


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="1",
            text="The quick brown fox jumps over the lazy dog",
            metadata={"source": "doc1"},
        ),
        Chunk(
            id="2",
            text="A fast brown fox leaps across a sleeping canine",
            metadata={"source": "doc2"},
        ),
        Chunk(
            id="3",
            text="Python is a high-level programming language",
            metadata={"source": "doc3"},
        ),
        Chunk(
            id="4",
            text="Machine learning algorithms require large datasets",
            metadata={"source": "doc4"},
        ),
    ]


@pytest.fixture
def bm25_retriever(sample_chunks):
    """Create and fit a BM25 retriever."""
    retriever = BM25Retriever()
    retriever.fit(sample_chunks)
    return retriever


def test_bm25_initialization():
    """Test BM25 retriever initialization."""
    retriever = BM25Retriever()
    assert retriever.name == "BM25"
    assert retriever.k1 == 1.5
    assert retriever.b == 0.75
    assert not retriever._is_fitted


def test_bm25_custom_parameters():
    """Test BM25 with custom k1 and b parameters."""
    retriever = BM25Retriever(k1=2.0, b=0.5)
    assert retriever.k1 == 2.0
    assert retriever.b == 0.5


def test_bm25_fit(sample_chunks):
    """Test fitting BM25 on corpus."""
    retriever = BM25Retriever()
    retriever.fit(sample_chunks)

    assert retriever._is_fitted
    assert retriever.bm25 is not None
    assert len(retriever.chunks) == len(sample_chunks)


def test_bm25_retrieve_before_fit():
    """Test that retrieving before fitting raises error."""
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="fit"):
        retriever.retrieve("test query")


def test_bm25_retrieve_basic(bm25_retriever):
    """Test basic BM25 retrieval."""
    result = bm25_retriever.retrieve("brown fox", k=2)

    assert len(result.chunks) == 2
    assert len(result.scores) == 2
    assert result.method == "BM25"
    assert result.latency_ms > 0

    # First result should have "brown fox" in text
    assert "brown" in result.chunks[0].text.lower()
    assert "fox" in result.chunks[0].text.lower()


def test_bm25_retrieve_with_k(bm25_retriever):
    """Test retrieval with different k values."""
    result_k1 = bm25_retriever.retrieve("programming language", k=1)
    result_k3 = bm25_retriever.retrieve("programming language", k=3)

    assert len(result_k1.chunks) == 1
    assert len(result_k3.chunks) == 3
    assert result_k1.chunks[0].id == result_k3.chunks[0].id  # Same top result


def test_bm25_scoring_relevance(bm25_retriever):
    """Test that BM25 scores are in descending order."""
    result = bm25_retriever.retrieve("fox dog", k=4)

    # Scores should be descending
    for i in range(len(result.scores) - 1):
        assert result.scores[i] >= result.scores[i + 1]


def test_bm25_no_results_for_empty_query(bm25_retriever):
    """Test behavior with empty query."""
    result = bm25_retriever.retrieve("", k=2)

    assert len(result.chunks) == 2  # Still returns chunks but with low scores
    assert all(score >= 0 for score in result.scores)


def test_bm25_exact_match_highest_score(bm25_retriever):
    """Test that exact matches get highest scores."""
    result = bm25_retriever.retrieve("Python is a high-level programming language", k=4)

    # Document 3 should be top result
    assert result.chunks[0].id == "3"
    assert result.scores[0] > result.scores[1]


def test_bm25_tokenization(sample_chunks):
    """Test custom tokenizer."""

    def custom_tokenizer(text):
        return text.lower().split()

    retriever = BM25Retriever(tokenizer=custom_tokenizer)
    retriever.fit(sample_chunks)

    result = retriever.retrieve("FOX", k=2)
    assert len(result.chunks) == 2
    assert "fox" in result.chunks[0].text.lower()


def test_bm25_metadata_preservation(bm25_retriever):
    """Test that chunk metadata is preserved."""
    result = bm25_retriever.retrieve("fox", k=2)

    for chunk in result.chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"].startswith("doc")


def test_bm25_result_metadata(bm25_retriever):
    """Test retrieval result metadata."""
    result = bm25_retriever.retrieve("test", k=2)

    assert "bm25_params" in result.metadata
    assert result.metadata["bm25_params"]["k1"] == 1.5
    assert result.metadata["bm25_params"]["b"] == 0.75


def test_bm25_refit(sample_chunks):
    """Test refitting with different corpus."""
    retriever = BM25Retriever()
    retriever.fit(sample_chunks[:2])

    result1 = retriever.retrieve("Python", k=2)

    # Refit with all chunks
    retriever.fit(sample_chunks)
    result2 = retriever.retrieve("Python", k=2)

    # Results should be different
    assert len(retriever.chunks) == 4
    assert result1.chunks[0].id != result2.chunks[0].id or len(result1.chunks) != len(
        result2.chunks
    )
