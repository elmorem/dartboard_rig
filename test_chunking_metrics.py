"""
Tests for chunking evaluation metrics.
"""

import pytest
import numpy as np
from dartboard.ingestion.chunking import Chunk
from dartboard.evaluation.chunking_metrics import (
    ChunkingEvaluator,
    ChunkingMetrics,
    evaluate_chunking_quality,
)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(text="First chunk with some text.", metadata={}, chunk_index=0),
        Chunk(text="Second chunk with more text here.", metadata={}, chunk_index=1),
        Chunk(text="Third chunk.", metadata={}, chunk_index=2),
        Chunk(
            text="Fourth chunk has the longest text in this set.",
            metadata={},
            chunk_index=3,
        ),
    ]


@pytest.fixture
def sample_document():
    """Sample original document."""
    return "First chunk with some text. Second chunk with more text here. Third chunk. Fourth chunk has the longest text in this set."


class TestChunkingEvaluator:
    """Tests for ChunkingEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return ChunkingEvaluator()

    def test_basic_metrics(self, evaluator, sample_chunks, sample_document):
        """Test basic size and distribution metrics."""
        metrics = evaluator.evaluate(sample_chunks, sample_document)

        assert metrics.num_chunks == 4
        assert metrics.avg_chunk_size > 0
        assert metrics.std_chunk_size >= 0
        assert metrics.min_chunk_size <= metrics.max_chunk_size
        assert 0 <= metrics.coverage_ratio <= 2.0  # Allow for overlap

    def test_empty_chunks(self, evaluator):
        """Test handling of empty chunk list."""
        metrics = evaluator.evaluate([], "some text")

        assert metrics.num_chunks == 0
        assert metrics.avg_chunk_size == 0
        assert metrics.coverage_ratio == 0.0

    def test_single_chunk(self, evaluator):
        """Test metrics for single chunk."""
        chunks = [Chunk(text="Only chunk", metadata={}, chunk_index=0)]
        document = "Only chunk"

        metrics = evaluator.evaluate(chunks, document)

        assert metrics.num_chunks == 1
        assert metrics.std_chunk_size == 0.0  # No variance with one chunk
        assert metrics.avg_overlap is None  # No overlap with single chunk

    def test_size_metrics(self, evaluator):
        """Test size-related metrics calculations."""
        chunks = [
            Chunk(text="a" * 100, metadata={}, chunk_index=0),
            Chunk(text="b" * 150, metadata={}, chunk_index=1),
            Chunk(text="c" * 200, metadata={}, chunk_index=2),
        ]

        metrics = evaluator.evaluate(chunks, "abc")

        assert metrics.avg_chunk_size == 150.0  # (100 + 150 + 200) / 3
        assert metrics.min_chunk_size == 100
        assert metrics.max_chunk_size == 200
        assert metrics.median_chunk_size == 150.0

    def test_coefficient_of_variation(self, evaluator):
        """Test CV calculation."""
        # Uniform sizes -> low CV
        uniform_chunks = [
            Chunk(text="x" * 100, metadata={}, chunk_index=i) for i in range(5)
        ]

        metrics_uniform = evaluator.evaluate(uniform_chunks, "x" * 500)
        assert metrics_uniform.size_coefficient_variation == 0.0

        # Variable sizes -> higher CV
        variable_chunks = [
            Chunk(text="a" * 50, metadata={}, chunk_index=0),
            Chunk(text="b" * 200, metadata={}, chunk_index=1),
        ]

        metrics_variable = evaluator.evaluate(variable_chunks, "ab")
        assert metrics_variable.size_coefficient_variation > 0

    def test_overlap_computation(self, evaluator):
        """Test overlap metrics."""
        # Chunks with exact overlap
        chunks = [
            Chunk(text="Hello world today", metadata={}, chunk_index=0),
            Chunk(text="today is sunny", metadata={}, chunk_index=1),
        ]

        metrics = evaluator.evaluate(chunks, "text")

        assert metrics.avg_overlap is not None
        assert metrics.avg_overlap >= 0

    def test_coherence_with_embeddings(self, evaluator):
        """Test coherence metrics with embeddings."""
        chunks = [
            Chunk(text="Text 1", metadata={}, chunk_index=0),
            Chunk(text="Text 2", metadata={}, chunk_index=1),
            Chunk(text="Text 3", metadata={}, chunk_index=2),
        ]

        # Create dummy embeddings (normalized)
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        metrics = evaluator.evaluate(chunks, "text", embeddings=embeddings)

        assert metrics.avg_coherence is not None
        assert metrics.min_coherence is not None
        assert 0 <= metrics.avg_coherence <= 1.0
        assert 0 <= metrics.min_coherence <= 1.0

    def test_compare_strategies(self, evaluator, sample_chunks, sample_document):
        """Test comparing multiple chunking strategies."""
        strategy1 = [
            Chunk(text="Short", metadata={}, chunk_index=0),
            Chunk(text="Chunks", metadata={}, chunk_index=1),
        ]

        strategy2 = [Chunk(text="One long chunk", metadata={}, chunk_index=0)]

        results = evaluator.compare_strategies(
            {
                "strategy1": (strategy1, sample_document),
                "strategy2": (strategy2, sample_document),
            }
        )

        assert "strategy1" in results
        assert "strategy2" in results
        assert results["strategy1"].num_chunks == 2
        assert results["strategy2"].num_chunks == 1

    def test_metrics_str_representation(
        self, evaluator, sample_chunks, sample_document
    ):
        """Test string representation of metrics."""
        metrics = evaluator.evaluate(sample_chunks, sample_document)

        str_repr = str(metrics)

        assert "Chunking Metrics" in str_repr
        assert "Total Chunks" in str_repr
        assert "Avg Size" in str_repr


def test_convenience_function(sample_chunks, sample_document):
    """Test convenience evaluation function."""
    metrics = evaluate_chunking_quality(sample_chunks, sample_document, verbose=False)

    assert isinstance(metrics, ChunkingMetrics)
    assert metrics.num_chunks == len(sample_chunks)


def test_coverage_ratio():
    """Test coverage ratio calculation."""
    evaluator = ChunkingEvaluator()

    # Exact coverage (no overlap)
    chunks = [
        Chunk(text="Hello ", metadata={}, chunk_index=0),
        Chunk(text="world", metadata={}, chunk_index=1),
    ]
    doc = "Hello world"

    metrics = evaluator.evaluate(chunks, doc)
    assert 0.9 <= metrics.coverage_ratio <= 1.1  # Allow small variance

    # Over-coverage (with overlap)
    chunks_overlap = [
        Chunk(text="Hello world", metadata={}, chunk_index=0),
        Chunk(text="world today", metadata={}, chunk_index=1),
    ]

    metrics_overlap = evaluator.evaluate(chunks_overlap, doc)
    assert metrics_overlap.coverage_ratio > 1.0  # Overlap increases coverage


def test_edge_cases():
    """Test various edge cases."""
    evaluator = ChunkingEvaluator()

    # Empty document - coverage should be 0 (dividing by 0 returns 0 in our impl)
    chunks = [Chunk(text="text", metadata={}, chunk_index=0)]
    metrics = evaluator.evaluate(chunks, "")
    assert metrics.coverage_ratio >= 0  # Should handle gracefully

    # Very long chunks
    long_chunk = Chunk(text="x" * 10000, metadata={}, chunk_index=0)
    metrics_long = evaluator.evaluate([long_chunk], "x" * 10000)
    assert metrics_long.avg_chunk_size == 10000


if __name__ == "__main__":
    print("Running chunking metrics tests...")
    pytest.main([__file__, "-v"])
