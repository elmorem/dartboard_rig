"""
Chunking quality evaluation metrics.

Provides metrics to assess and compare chunking strategies:
- Statistical metrics (size, variance, distribution)
- Semantic coherence (within-chunk similarity)
- Context preservation (between-chunk overlap)
- Coverage metrics (information retention)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dartboard.ingestion.chunking import Chunk


@dataclass
class ChunkingMetrics:
    """Container for chunking evaluation metrics."""

    # Size metrics
    num_chunks: int
    avg_chunk_size: float
    std_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    median_chunk_size: float

    # Distribution metrics
    size_variance: float
    size_coefficient_variation: float  # std / mean

    # Coverage metrics
    total_chars: int
    coverage_ratio: float  # chars in chunks / original doc chars

    # Overlap metrics (if applicable)
    avg_overlap: Optional[float] = None

    # Semantic metrics (if embeddings provided)
    avg_coherence: Optional[float] = None
    min_coherence: Optional[float] = None

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Chunking Metrics ===",
            f"Total Chunks: {self.num_chunks}",
            f"Avg Size: {self.avg_chunk_size:.1f} chars",
            f"Std Dev: {self.std_chunk_size:.1f}",
            f"Range: [{self.min_chunk_size}, {self.max_chunk_size}]",
            f"CV: {self.size_coefficient_variation:.2f}",
            f"Coverage: {self.coverage_ratio:.1%}",
        ]

        if self.avg_overlap is not None:
            lines.append(f"Avg Overlap: {self.avg_overlap:.1f} chars")

        if self.avg_coherence is not None:
            lines.append(f"Avg Coherence: {self.avg_coherence:.3f}")
            lines.append(f"Min Coherence: {self.min_coherence:.3f}")

        return "\n".join(lines)


class ChunkingEvaluator:
    """Evaluate chunking quality with various metrics."""

    def evaluate(
        self,
        chunks: List[Chunk],
        original_text: str,
        embeddings: Optional[np.ndarray] = None,
    ) -> ChunkingMetrics:
        """
        Compute comprehensive metrics for chunked document.

        Args:
            chunks: List of chunks
            original_text: Original document text
            embeddings: Optional chunk embeddings for coherence metrics

        Returns:
            ChunkingMetrics object
        """
        if not chunks:
            return ChunkingMetrics(
                num_chunks=0,
                avg_chunk_size=0.0,
                std_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                median_chunk_size=0.0,
                size_variance=0.0,
                size_coefficient_variation=0.0,
                total_chars=0,
                coverage_ratio=0.0,
            )

        # Size metrics
        sizes = [len(chunk.text) for chunk in chunks]

        num_chunks = len(chunks)
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        median_size = np.median(sizes)
        variance = np.var(sizes)
        cv = std_size / avg_size if avg_size > 0 else 0.0

        # Coverage metrics
        total_chars = sum(sizes)
        coverage = total_chars / len(original_text) if len(original_text) > 0 else 0.0

        # Overlap metrics
        avg_overlap = self._compute_overlap(chunks) if len(chunks) > 1 else None

        # Semantic metrics
        avg_coherence, min_coherence = None, None
        if embeddings is not None and len(embeddings) > 0:
            avg_coherence, min_coherence = self._compute_coherence(embeddings)

        return ChunkingMetrics(
            num_chunks=num_chunks,
            avg_chunk_size=avg_size,
            std_chunk_size=std_size,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            median_chunk_size=median_size,
            size_variance=variance,
            size_coefficient_variation=cv,
            total_chars=total_chars,
            coverage_ratio=coverage,
            avg_overlap=avg_overlap,
            avg_coherence=avg_coherence,
            min_coherence=min_coherence,
        )

    def _compute_overlap(self, chunks: List[Chunk]) -> float:
        """
        Compute average character overlap between consecutive chunks.

        Args:
            chunks: List of chunks

        Returns:
            Average overlap in characters
        """
        overlaps = []

        for i in range(len(chunks) - 1):
            current = chunks[i].text
            next_chunk = chunks[i + 1].text

            # Find longest common suffix/prefix
            overlap = 0
            max_check = min(len(current), len(next_chunk), 500)  # Limit search

            for length in range(1, max_check + 1):
                if current[-length:] == next_chunk[:length]:
                    overlap = length
                else:
                    break

            overlaps.append(overlap)

        return np.mean(overlaps) if overlaps else 0.0

    def _compute_coherence(
        self, embeddings: np.ndarray
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute semantic coherence from embeddings.

        Coherence = average cosine similarity of each chunk to its neighbors.

        Args:
            embeddings: Chunk embeddings (N x D)

        Returns:
            (avg_coherence, min_coherence) tuple
        """
        if len(embeddings) < 2:
            return None, None

        coherences = []

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        # Compute similarity to neighbors
        for i in range(len(embeddings)):
            neighbors = []

            # Previous chunk
            if i > 0:
                neighbors.append(i - 1)

            # Next chunk
            if i < len(embeddings) - 1:
                neighbors.append(i + 1)

            if neighbors:
                similarities = [np.dot(normalized[i], normalized[j]) for j in neighbors]
                coherences.append(np.mean(similarities))

        if coherences:
            return float(np.mean(coherences)), float(np.min(coherences))
        else:
            return None, None

    def compare_strategies(
        self,
        strategy_results: Dict[str, tuple[List[Chunk], str]],
        embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, ChunkingMetrics]:
        """
        Compare multiple chunking strategies on the same document.

        Args:
            strategy_results: Dict of {strategy_name: (chunks, original_text)}
            embeddings_dict: Optional dict of {strategy_name: embeddings}

        Returns:
            Dict of {strategy_name: ChunkingMetrics}
        """
        results = {}

        for strategy_name, (chunks, original_text) in strategy_results.items():
            embeddings = embeddings_dict.get(strategy_name) if embeddings_dict else None

            metrics = self.evaluate(chunks, original_text, embeddings)
            results[strategy_name] = metrics

        return results

    def print_comparison(self, metrics_dict: Dict[str, ChunkingMetrics]) -> None:
        """
        Print side-by-side comparison of chunking strategies.

        Args:
            metrics_dict: Dict of {strategy_name: metrics}
        """
        print("=" * 80)
        print("CHUNKING STRATEGY COMPARISON")
        print("=" * 80)

        # Header
        strategies = list(metrics_dict.keys())
        print(f"\n{'Metric':<25} " + " ".join([f"{s:>15}" for s in strategies]))
        print("-" * 80)

        # Metrics to compare
        comparisons = [
            ("Num Chunks", lambda m: f"{m.num_chunks:>15}"),
            ("Avg Size (chars)", lambda m: f"{m.avg_chunk_size:>15.1f}"),
            ("Std Dev", lambda m: f"{m.std_chunk_size:>15.1f}"),
            ("CV", lambda m: f"{m.size_coefficient_variation:>15.3f}"),
            ("Coverage", lambda m: f"{m.coverage_ratio:>14.1%}"),
        ]

        # Add overlap if available
        if any(m.avg_overlap is not None for m in metrics_dict.values()):
            comparisons.append(
                ("Avg Overlap", lambda m: f"{m.avg_overlap or 0:>15.1f}")
            )

        # Add coherence if available
        if any(m.avg_coherence is not None for m in metrics_dict.values()):
            comparisons.append(
                ("Avg Coherence", lambda m: f"{m.avg_coherence or 0:>15.3f}")
            )
            comparisons.append(
                ("Min Coherence", lambda m: f"{m.min_coherence or 0:>15.3f}")
            )

        # Print each metric
        for metric_name, formatter in comparisons:
            values = [formatter(metrics_dict[s]) for s in strategies]
            print(f"{metric_name:<25} " + " ".join(values))

        print("=" * 80)


def evaluate_chunking_quality(
    chunks: List[Chunk], original_text: str, verbose: bool = True
) -> ChunkingMetrics:
    """
    Convenience function to evaluate chunking quality.

    Args:
        chunks: List of chunks
        original_text: Original document text
        verbose: Print summary if True

    Returns:
        ChunkingMetrics object
    """
    evaluator = ChunkingEvaluator()
    metrics = evaluator.evaluate(chunks, original_text)

    if verbose:
        print(metrics)

    return metrics
