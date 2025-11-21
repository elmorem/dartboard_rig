from __future__ import annotations

"""
Utility functions for Dartboard RAG algorithm.

Includes log-space operations, similarity metrics, and numerical stability helpers.
"""

import numpy as np
from scipy.special import logsumexp
from typing import Union, Optional


def log_gaussian_kernel(
    a: np.ndarray, b: np.ndarray, sigma: float, eps: float = 1e-10
) -> float:
    """
    Compute log of Gaussian kernel N(a, b, σ).

    Formula: -log(σ) - 0.5*log(2π) - ||a-b||²/(2σ²)

    Args:
        a: First embedding vector
        b: Second embedding vector
        sigma: Temperature parameter
        eps: Small constant for numerical stability

    Returns:
        Log-probability value

    Raises:
        ValueError: If embeddings have different dimensions

    Example:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.1, 2.1, 3.1])
        >>> log_prob = log_gaussian_kernel(a, b, sigma=1.0)
        >>> log_prob < 0  # Log probabilities are negative
        True
    """
    if a.shape != b.shape:
        raise ValueError(f"Embedding dimensions must match: {a.shape} vs {b.shape}")

    # Ensure sigma is not too small to prevent division by zero
    sigma = max(sigma, eps)

    # Compute squared Euclidean distance
    squared_dist = np.sum((a - b) ** 2)

    # Log of normalization constant
    log_normalizer = -np.log(sigma) - 0.5 * np.log(2 * np.pi)

    # Full log-probability
    log_kernel = log_normalizer - squared_dist / (2 * sigma**2)

    return float(log_kernel)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: cos(θ) = (a · b) / (||a|| × ||b||)

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]

    Example:
        >>> a = np.array([1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(a, b)
        1.0
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance: 1 - cosine_similarity(a, b).

    Range: [0, 2], where 0 = identical, 2 = opposite

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine distance

    Example:
        >>> a = np.array([1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 0.0])
        >>> cosine_distance(a, b)
        0.0
    """
    return 1.0 - cosine_similarity(a, b)


def logsumexp_stable(
    log_values: np.ndarray, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Numerically stable log-sum-exp.

    Computes: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))

    Args:
        log_values: Array of log-probability values
        axis: Axis along which to compute (None for all)

    Returns:
        Log of the sum of exponentials

    Example:
        >>> log_vals = np.array([-1000.0, -1000.0, -1000.0])
        >>> result = logsumexp_stable(log_vals)
        >>> np.isfinite(result)
        True
    """
    return logsumexp(log_values, axis=axis)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length (L2 normalization).

    Args:
        embeddings: Array of shape (n, dim) or (dim,)

    Returns:
        Normalized embeddings of same shape

    Example:
        >>> embeddings = np.array([[3.0, 4.0]])
        >>> normalized = normalize_embeddings(embeddings)
        >>> np.allclose(np.linalg.norm(normalized[0]), 1.0)
        True
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        if norm == 0:
            return embeddings
        return embeddings / norm

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Prevent division by zero
    return embeddings / norms


def batch_cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple documents efficiently.

    Args:
        query: Single embedding of shape (dim,)
        documents: Multiple embeddings of shape (n, dim)

    Returns:
        Array of similarities of shape (n,)

    Example:
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> sims = batch_cosine_similarity(query, docs)
        >>> len(sims) == 2
        True
    """
    # Normalize
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    docs_norm = normalize_embeddings(documents)

    # Dot product
    similarities = np.dot(docs_norm, query_norm)

    return similarities


def clamp_similarity(
    similarity: float, min_val: float = -1.0, max_val: float = 1.0
) -> float:
    """
    Clamp similarity values to valid range.

    Args:
        similarity: Similarity value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped similarity

    Example:
        >>> clamp_similarity(1.5)
        1.0
        >>> clamp_similarity(-1.5)
        -1.0
    """
    return max(min_val, min(max_val, similarity))
