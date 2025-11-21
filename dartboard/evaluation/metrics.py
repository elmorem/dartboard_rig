from __future__ import annotations

"""
Evaluation metrics for Dartboard RAG retrieval.

Implements standard IR metrics plus diversity measures:
- NDCG (Normalized Discounted Cumulative Gain)
- Precision@K / Recall@K
- Mean Average Precision (MAP)
- Diversity metrics (intra-list distance, coverage)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from dartboard.datasets.models import Chunk, RetrievalResult, Dataset
from dartboard.utils import cosine_distance


@dataclass
class MetricResult:
    """Container for evaluation metric results."""

    metric_name: str
    value: float
    metadata: Dict[str, Any]


@dataclass
class EvaluationReport:
    """Complete evaluation report for a retrieval system."""

    dataset_name: str
    metrics: Dict[str, float]
    per_query_metrics: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"Evaluation Report: {self.dataset_name}", "=" * 50]
        for metric, value in self.metrics.items():
            lines.append(f"{metric:30s}: {value:.4f}")
        return "\n".join(lines)


class RetrievalEvaluator:
    """Evaluates retrieval system performance."""

    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize evaluator.

        Args:
            k_values: List of K values for Precision@K and Recall@K
        """
        self.k_values = k_values

    def evaluate_dataset(
        self, results: List[RetrievalResult], dataset: Dataset
    ) -> EvaluationReport:
        """
        Evaluate retrieval results against ground truth.

        Args:
            results: List of retrieval results (one per query)
            dataset: Dataset with ground truth

        Returns:
            Complete evaluation report
        """
        if len(results) != len(dataset.queries):
            raise ValueError(
                f"Number of results ({len(results)}) must match queries ({len(dataset.queries)})"
            )

        per_query_metrics = {}
        aggregate_metrics = {
            "ndcg": [],
            "map": [],
            "diversity": [],
        }

        # Add precision/recall for each K
        for k in self.k_values:
            aggregate_metrics[f"precision@{k}"] = []
            aggregate_metrics[f"recall@{k}"] = []

        # Evaluate each query
        for i, (result, query) in enumerate(zip(results, dataset.queries)):
            query_id = f"query_{i}"
            ground_truth = dataset.ground_truth.get(query_id, [])

            # Compute metrics
            metrics = {}
            metrics["ndcg"] = self.compute_ndcg(result, ground_truth)
            metrics["diversity"] = self.compute_diversity(result)

            for k in self.k_values:
                p, r = self.compute_precision_recall(result, ground_truth, k)
                metrics[f"precision@{k}"] = p
                metrics[f"recall@{k}"] = r

            # Store per-query metrics
            per_query_metrics[query_id] = metrics

            # Aggregate
            for key, value in metrics.items():
                if key in aggregate_metrics:
                    aggregate_metrics[key].append(value)

        # Compute MAP separately
        aggregate_metrics["map"] = [
            self.compute_average_precision(
                results[i], dataset.ground_truth.get(f"query_{i}", [])
            )
            for i in range(len(results))
        ]

        # Average all metrics
        averaged_metrics = {
            key: np.mean(values) for key, values in aggregate_metrics.items()
        }

        return EvaluationReport(
            dataset_name=dataset.name,
            metrics=averaged_metrics,
            per_query_metrics=per_query_metrics,
            metadata={"num_queries": len(results), "k_values": self.k_values},
        )

    def compute_ndcg(
        self, result: RetrievalResult, ground_truth: List[str], k: Optional[int] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.

        Args:
            result: Retrieval result
            ground_truth: List of relevant chunk IDs
            k: Optional cutoff (uses all results if None)

        Returns:
            NDCG score in [0, 1]
        """
        if not ground_truth:
            return 0.0

        retrieved_ids = [chunk.id for chunk in result.chunks]
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        # Compute DCG
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved_ids):
            relevance = 1.0 if chunk_id in ground_truth else 0.0
            dcg += relevance / np.log2(i + 2)  # i+2 because i is 0-indexed

        # Compute IDCG (ideal DCG)
        ideal_relevance = [1.0] * min(len(ground_truth), len(retrieved_ids))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def compute_precision_recall(
        self, result: RetrievalResult, ground_truth: List[str], k: int
    ) -> Tuple[float, float]:
        """
        Compute Precision@K and Recall@K.

        Args:
            result: Retrieval result
            ground_truth: List of relevant chunk IDs
            k: Cutoff value

        Returns:
            (precision, recall) tuple
        """
        if not ground_truth:
            return 0.0, 0.0

        retrieved_ids = [chunk.id for chunk in result.chunks[:k]]
        relevant_retrieved = len(set(retrieved_ids) & set(ground_truth))

        precision = relevant_retrieved / k if k > 0 else 0.0
        recall = relevant_retrieved / len(ground_truth) if ground_truth else 0.0

        return precision, recall

    def compute_average_precision(
        self, result: RetrievalResult, ground_truth: List[str]
    ) -> float:
        """
        Compute Average Precision (AP).

        Args:
            result: Retrieval result
            ground_truth: List of relevant chunk IDs

        Returns:
            AP score
        """
        if not ground_truth:
            return 0.0

        retrieved_ids = [chunk.id for chunk in result.chunks]
        precisions = []
        num_relevant = 0

        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in ground_truth:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precisions.append(precision_at_i)

        if not precisions:
            return 0.0

        return sum(precisions) / len(ground_truth)

    def compute_diversity(self, result: RetrievalResult) -> float:
        """
        Compute diversity score (average pairwise distance).

        Args:
            result: Retrieval result

        Returns:
            Diversity score (higher = more diverse)
        """
        if len(result.chunks) < 2:
            return 0.0

        embeddings = [chunk.embedding for chunk in result.chunks]
        distances = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine_distance(embeddings[i], embeddings[j])
                distances.append(dist)

        return float(np.mean(distances))

    def compute_coverage(
        self, result: RetrievalResult, ground_truth_clusters: Dict[str, int]
    ) -> float:
        """
        Compute cluster coverage (what fraction of clusters are represented).

        Args:
            result: Retrieval result
            ground_truth_clusters: Mapping of chunk_id -> cluster_id

        Returns:
            Coverage score in [0, 1]
        """
        if not ground_truth_clusters:
            return 0.0

        retrieved_ids = [chunk.id for chunk in result.chunks]
        retrieved_clusters = set(
            ground_truth_clusters.get(chunk_id)
            for chunk_id in retrieved_ids
            if chunk_id in ground_truth_clusters
        )
        retrieved_clusters.discard(None)

        all_clusters = set(ground_truth_clusters.values())
        if not all_clusters:
            return 0.0

        return len(retrieved_clusters) / len(all_clusters)


class ComparisonEvaluator:
    """Compare multiple retrieval systems."""

    def __init__(self):
        """Initialize comparison evaluator."""
        self.evaluator = RetrievalEvaluator()

    def compare_systems(
        self,
        system_results: Dict[str, List[RetrievalResult]],
        dataset: Dataset,
    ) -> Dict[str, EvaluationReport]:
        """
        Compare multiple systems on the same dataset.

        Args:
            system_results: Dict of system_name -> list of results
            dataset: Evaluation dataset

        Returns:
            Dict of system_name -> evaluation report
        """
        reports = {}
        for system_name, results in system_results.items():
            reports[system_name] = self.evaluator.evaluate_dataset(results, dataset)
        return reports

    def generate_comparison_table(self, reports: Dict[str, EvaluationReport]) -> str:
        """
        Generate comparison table.

        Args:
            reports: Dict of system_name -> report

        Returns:
            Formatted table string
        """
        if not reports:
            return "No reports to compare"

        # Get all metrics
        all_metrics = set()
        for report in reports.values():
            all_metrics.update(report.metrics.keys())

        # Sort metrics
        metric_order = ["ndcg", "map", "diversity"]
        sorted_metrics = sorted(
            all_metrics,
            key=lambda x: (
                metric_order.index(x) if x in metric_order else 999,
                x,
            ),
        )

        # Build table
        lines = ["System Comparison", "=" * 80]

        # Header
        header = f"{'Metric':<20}"
        for system_name in reports.keys():
            header += f"{system_name:>15}"
        lines.append(header)
        lines.append("-" * 80)

        # Rows
        for metric in sorted_metrics:
            row = f"{metric:<20}"
            for report in reports.values():
                value = report.metrics.get(metric, 0.0)
                row += f"{value:>15.4f}"
            lines.append(row)

        return "\n".join(lines)


class DiversityAnalyzer:
    """Analyze diversity properties of retrieved results."""

    def analyze_diversity(
        self, result: RetrievalResult, num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze diversity characteristics.

        Args:
            result: Retrieval result
            num_bins: Number of bins for distance histogram

        Returns:
            Dictionary with diversity statistics
        """
        if len(result.chunks) < 2:
            return {"error": "Need at least 2 chunks for diversity analysis"}

        embeddings = [chunk.embedding for chunk in result.chunks]
        distances = []

        # Compute all pairwise distances
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine_distance(embeddings[i], embeddings[j])
                distances.append(dist)

        distances = np.array(distances)

        return {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "median_distance": float(np.median(distances)),
            "num_pairs": len(distances),
            "histogram": np.histogram(distances, bins=num_bins)[0].tolist(),
        }
