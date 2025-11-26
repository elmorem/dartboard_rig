"""
Evaluation metrics for retrieval systems.

Implements standard Information Retrieval metrics:
- Mean Reciprocal Rank (MRR@K)
- Recall@K
- Precision@K
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP)
"""

import logging
from typing import List, Dict, Set, Any
import numpy as np

logger = logging.getLogger(__name__)


def mean_reciprocal_rank(
    results: List[str], relevant_docs: Set[str], k: int = 10
) -> float:
    """
    Calculate Mean Reciprocal Rank at K.

    MRR@K = 1 / rank of first relevant document (if within top K)

    Args:
        results: List of retrieved document IDs (in ranked order)
        relevant_docs: Set of relevant document IDs
        k: Cutoff rank (default: 10)

    Returns:
        MRR score (0.0 to 1.0)

    Example:
        >>> results = ["doc1", "doc2", "doc3"]
        >>> relevant = {"doc2"}
        >>> mean_reciprocal_rank(results, relevant, k=10)
        0.5  # First relevant doc at rank 2, so 1/2 = 0.5
    """
    if not results or not relevant_docs:
        return 0.0

    # Check only top-k results
    for rank, doc_id in enumerate(results[:k], 1):
        if doc_id in relevant_docs:
            return 1.0 / rank

    return 0.0


def recall_at_k(results: List[str], relevant_docs: Set[str], k: int = 10) -> float:
    """
    Calculate Recall at K.

    Recall@K = (# relevant docs in top K) / (total # relevant docs)

    Args:
        results: List of retrieved document IDs (in ranked order)
        relevant_docs: Set of relevant document IDs
        k: Cutoff rank (default: 10)

    Returns:
        Recall score (0.0 to 1.0)

    Example:
        >>> results = ["doc1", "doc2", "doc3"]
        >>> relevant = {"doc2", "doc4"}
        >>> recall_at_k(results, relevant, k=10)
        0.5  # 1 out of 2 relevant docs found
    """
    if not relevant_docs:
        return 0.0

    if not results:
        return 0.0

    # Get top-k results
    top_k = set(results[:k])

    # Count relevant docs in top-k
    relevant_in_top_k = top_k & relevant_docs

    return len(relevant_in_top_k) / len(relevant_docs)


def precision_at_k(results: List[str], relevant_docs: Set[str], k: int = 10) -> float:
    """
    Calculate Precision at K.

    Precision@K = (# relevant docs in top K) / K

    Args:
        results: List of retrieved document IDs (in ranked order)
        relevant_docs: Set of relevant document IDs
        k: Cutoff rank (default: 10)

    Returns:
        Precision score (0.0 to 1.0)

    Example:
        >>> results = ["doc1", "doc2", "doc3"]
        >>> relevant = {"doc2"}
        >>> precision_at_k(results, relevant, k=3)
        0.333  # 1 out of 3 docs is relevant
    """
    if not results:
        return 0.0

    # Get top-k results (or all if fewer than k)
    top_k = results[:k]

    if not top_k:
        return 0.0

    # Count relevant docs in top-k
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)

    return relevant_in_top_k / len(top_k)


def dcg_at_k(
    results: List[str],
    relevant_docs: Set[str],
    k: int = 10,
    gains: Dict[str, float] = None,
) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    DCG@K = sum_{i=1}^K (gain_i / log2(i + 1))

    Args:
        results: List of retrieved document IDs (in ranked order)
        relevant_docs: Set of relevant document IDs
        k: Cutoff rank (default: 10)
        gains: Optional dict mapping doc_id to relevance gain (default: binary 1.0)

    Returns:
        DCG score
    """
    if not results:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(results[:k], 1):
        if doc_id in relevant_docs:
            # Use provided gain or default to 1.0 for binary relevance
            gain = gains.get(doc_id, 1.0) if gains else 1.0
            dcg += gain / np.log2(rank + 1)

    return dcg


def ndcg_at_k(
    results: List[str],
    relevant_docs: Set[str],
    k: int = 10,
    gains: Dict[str, float] = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    where IDCG@K is the ideal DCG (if all relevant docs were at top).

    Args:
        results: List of retrieved document IDs (in ranked order)
        relevant_docs: Set of relevant document IDs
        k: Cutoff rank (default: 10)
        gains: Optional dict mapping doc_id to relevance gain (default: binary 1.0)

    Returns:
        NDCG score (0.0 to 1.0)

    Example:
        >>> results = ["doc1", "doc2", "doc3"]
        >>> relevant = {"doc2", "doc3"}
        >>> ndcg_at_k(results, relevant, k=10)
        0.785  # Example score
    """
    if not relevant_docs:
        return 0.0

    # Calculate actual DCG
    actual_dcg = dcg_at_k(results, relevant_docs, k, gains)

    # Calculate ideal DCG (all relevant docs ranked first)
    if gains:
        # Sort relevant docs by gain (descending)
        ideal_ranking = sorted(
            relevant_docs, key=lambda doc_id: gains.get(doc_id, 1.0), reverse=True
        )
    else:
        # For binary relevance, any ordering of relevant docs is ideal
        ideal_ranking = list(relevant_docs)

    ideal_dcg = dcg_at_k(ideal_ranking, relevant_docs, k, gains)

    # Normalize
    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def average_precision(
    results: List[str], relevant_docs: Set[str], k: int = None
) -> float:
    """
    Calculate Average Precision.

    AP = (1 / # relevant docs) * sum of (Precision@i * is_relevant_i)

    Args:
        results: List of retrieved document IDs (in ranked order)
        relevant_docs: Set of relevant document IDs
        k: Optional cutoff rank (default: all results)

    Returns:
        Average Precision score (0.0 to 1.0)
    """
    if not relevant_docs or not results:
        return 0.0

    # Limit to top-k if specified
    results_to_check = results[:k] if k else results

    num_relevant = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(results_to_check, 1):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / rank

    if num_relevant == 0:
        return 0.0

    return precision_sum / len(relevant_docs)


def evaluate_retrieval(
    results: List[str],
    relevant_docs: Set[str],
    k_values: List[int] = [1, 5, 10, 20, 100],
    gains: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval results with multiple metrics.

    Args:
        results: List of retrieved document IDs (in ranked order)
        relevant_docs: Set of relevant document IDs
        k_values: List of K values to compute metrics at
        gains: Optional dict mapping doc_id to relevance gain

    Returns:
        Dict of metric names to scores

    Example:
        >>> results = ["doc1", "doc2", "doc3", "doc4"]
        >>> relevant = {"doc2", "doc4"}
        >>> evaluate_retrieval(results, relevant, k_values=[1, 5, 10])
        {
            'MRR@10': 0.5,
            'Recall@1': 0.0,
            'Recall@5': 1.0,
            'Precision@1': 0.0,
            'Precision@5': 0.4,
            'NDCG@1': 0.0,
            'NDCG@5': 0.756,
            'AP': 0.625
        }
    """
    metrics = {}

    # MRR (typically computed at max K)
    max_k = max(k_values) if k_values else 10
    metrics[f"MRR@{max_k}"] = mean_reciprocal_rank(results, relevant_docs, k=max_k)

    # Compute metrics at each K value
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_at_k(results, relevant_docs, k=k)
        metrics[f"Precision@{k}"] = precision_at_k(results, relevant_docs, k=k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(results, relevant_docs, k=k, gains=gains)

    # Average Precision (no K cutoff)
    metrics["AP"] = average_precision(results, relevant_docs)

    return metrics


def mean_average_precision(
    all_results: List[List[str]], all_relevant_docs: List[Set[str]], k: int = None
) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.

    MAP = mean of Average Precision scores across all queries

    Args:
        all_results: List of result lists (one per query)
        all_relevant_docs: List of relevant doc sets (one per query)
        k: Optional cutoff rank

    Returns:
        MAP score (0.0 to 1.0)
    """
    if not all_results or not all_relevant_docs:
        return 0.0

    if len(all_results) != len(all_relevant_docs):
        raise ValueError(
            "Number of result lists must match number of relevant doc sets"
        )

    ap_scores = [
        average_precision(results, relevant_docs, k=k)
        for results, relevant_docs in zip(all_results, all_relevant_docs)
    ]

    return np.mean(ap_scores)


def evaluate_batch(
    all_results: List[List[str]],
    all_relevant_docs: List[Set[str]],
    k_values: List[int] = [1, 5, 10, 20, 100],
) -> Dict[str, float]:
    """
    Evaluate multiple queries and compute average metrics.

    Args:
        all_results: List of result lists (one per query)
        all_relevant_docs: List of relevant doc sets (one per query)
        k_values: List of K values to compute metrics at

    Returns:
        Dict of metric names to average scores across queries
    """
    if not all_results or not all_relevant_docs:
        return {}

    if len(all_results) != len(all_relevant_docs):
        raise ValueError(
            "Number of result lists must match number of relevant doc sets"
        )

    # Collect metrics for each query
    all_metrics = [
        evaluate_retrieval(results, relevant_docs, k_values)
        for results, relevant_docs in zip(all_results, all_relevant_docs)
    ]

    # Average across queries
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        scores = [m[metric_name] for m in all_metrics]
        avg_metrics[metric_name] = np.mean(scores)

    # Add MAP
    max_k = max(k_values) if k_values else None
    avg_metrics[f"MAP@{max_k}"] = mean_average_precision(
        all_results, all_relevant_docs, k=max_k
    )

    return avg_metrics


def pairwise_cosine_diversity(embeddings: np.ndarray) -> float:
    """
    Calculate average pairwise cosine distance (1 - similarity) between result embeddings.

    Higher values = more diverse results.

    Args:
        embeddings: (N, D) array of result embeddings

    Returns:
        Average pairwise cosine distance (0.0 to 1.0)
    """
    if len(embeddings) < 2:
        return 0.0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute pairwise cosine similarities
    similarities = np.dot(normalized, normalized.T)

    # Get upper triangle (excluding diagonal) for pairwise distances
    n = len(embeddings)
    distances = 1.0 - similarities

    # Average of upper triangle
    upper_triangle_indices = np.triu_indices(n, k=1)
    avg_distance = np.mean(distances[upper_triangle_indices])

    return float(avg_distance)


def intra_list_diversity(
    results: List[str], embeddings_dict: Dict[str, np.ndarray]
) -> float:
    """
    Calculate Intra-List Diversity (ILD) - average dissimilarity between all pairs.

    ILD = (1 / (|R| * (|R| - 1))) * sum_{i,j in R, i!=j} dissimilarity(i, j)

    Args:
        results: List of retrieved document IDs
        embeddings_dict: Dict mapping doc_id to embedding vector

    Returns:
        ILD score (0.0 to 1.0, higher = more diverse)
    """
    if len(results) < 2:
        return 0.0

    # Get embeddings for results
    result_embeddings = []
    for doc_id in results:
        if doc_id in embeddings_dict:
            result_embeddings.append(embeddings_dict[doc_id])

    if len(result_embeddings) < 2:
        return 0.0

    embeddings = np.array(result_embeddings)
    return pairwise_cosine_diversity(embeddings)


def subtopic_coverage(results: List[str], doc_subtopics: Dict[str, Set[str]]) -> float:
    """
    Calculate coverage of different subtopics in results.

    Measures how many unique subtopics are covered by the result set.

    Args:
        results: List of retrieved document IDs
        doc_subtopics: Dict mapping doc_id to set of subtopic labels

    Returns:
        Fraction of unique subtopics covered (0.0 to 1.0)
    """
    if not results or not doc_subtopics:
        return 0.0

    # Get all possible subtopics
    all_subtopics = set()
    for subtopics in doc_subtopics.values():
        all_subtopics.update(subtopics)

    if not all_subtopics:
        return 0.0

    # Get subtopics covered by results
    covered_subtopics = set()
    for doc_id in results:
        if doc_id in doc_subtopics:
            covered_subtopics.update(doc_subtopics[doc_id])

    return len(covered_subtopics) / len(all_subtopics)


def alpha_ndcg(
    results: List[str],
    relevant_docs: Set[str],
    embeddings_dict: Dict[str, np.ndarray],
    k: int = 10,
    alpha: float = 0.5,
) -> float:
    """
    Calculate alpha-NDCG which balances relevance and diversity.

    α-NDCG penalizes redundant documents (similar to already-seen docs).

    Args:
        results: List of retrieved document IDs
        relevant_docs: Set of relevant document IDs
        embeddings_dict: Dict mapping doc_id to embedding
        k: Cutoff rank
        alpha: Diversity weight (0 = pure relevance, 1 = pure diversity)

    Returns:
        α-NDCG score (0.0 to 1.0)
    """
    if not results or not relevant_docs:
        return 0.0

    top_k = results[:k]

    # Calculate α-DCG
    dcg = 0.0
    seen_embeddings = []

    for rank, doc_id in enumerate(top_k, 1):
        # Relevance gain
        rel = 1.0 if doc_id in relevant_docs else 0.0

        # Diversity penalty (novelty with respect to seen docs)
        novelty = 1.0
        if seen_embeddings and doc_id in embeddings_dict:
            current_emb = embeddings_dict[doc_id]
            # Compute max similarity to any seen doc
            max_sim = 0.0
            for seen_emb in seen_embeddings:
                sim = np.dot(current_emb, seen_emb) / (
                    np.linalg.norm(current_emb) * np.linalg.norm(seen_emb) + 1e-8
                )
                max_sim = max(max_sim, sim)
            novelty = 1.0 - max_sim

        # Combined gain
        gain = rel * (alpha + (1 - alpha) * novelty)
        dcg += gain / np.log2(rank + 1)

        # Track seen embeddings
        if doc_id in embeddings_dict:
            seen_embeddings.append(embeddings_dict[doc_id])

    # Calculate ideal α-DCG
    ideal_dcg = 0.0
    for rank in range(1, min(len(relevant_docs) + 1, k + 1)):
        # In ideal case, all docs are relevant and maximally diverse
        gain = 1.0  # Pure relevance
        ideal_dcg += gain / np.log2(rank + 1)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg
