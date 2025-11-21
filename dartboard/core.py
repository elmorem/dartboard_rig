from __future__ import annotations

"""
Core Dartboard RAG algorithm implementation.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import numpy as np
from dartboard.utils import (
    log_gaussian_kernel,
    logsumexp_stable,
    batch_cosine_similarity,
    normalize_embeddings,
)
from dartboard.datasets.models import Chunk, RetrievalResult
from dartboard.embeddings import EmbeddingModel, CrossEncoder


@dataclass
class DartboardConfig:
    """Configuration for Dartboard retrieval."""

    sigma: float = 1.0  # Temperature parameter
    top_k: int = 5  # Number of passages to retrieve
    triage_k: int = 100  # Number of candidates from triage
    reranker_type: Literal["cosine", "crossencoder", "hybrid"] = "hybrid"
    sigma_min: float = 1e-5  # Minimum sigma
    log_eps: float = 1e-10  # Epsilon for log stability


class DartboardRetriever:
    """Main retrieval class implementing Dartboard algorithm."""

    def __init__(
        self,
        config: DartboardConfig,
        embedding_model: EmbeddingModel,
        cross_encoder: Optional[CrossEncoder] = None,
    ):
        """
        Initialize Dartboard retriever.

        Args:
            config: Dartboard configuration
            embedding_model: Model for generating embeddings
            cross_encoder: Optional cross-encoder for reranking
        """
        self.config = config
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder

        # Ensure sigma is valid
        if self.config.sigma < self.config.sigma_min:
            self.config.sigma = self.config.sigma_min

    def retrieve(
        self, query: str, corpus: List[Chunk], return_scores: bool = False
    ) -> RetrievalResult:
        """
        Retrieve top-k chunks using Dartboard algorithm.

        Args:
            query: User query string
            corpus: List of document chunks to search
            return_scores: If True, include Dartboard scores

        Returns:
            RetrievalResult with top-k chunks and scores

        Raises:
            ValueError: If query is empty or corpus is invalid
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not corpus:
            raise ValueError("Corpus cannot be empty")

        # Encode query
        query_embedding = self.embedding_model.encode(query)
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0]

        # Triage: Get top candidates using KNN
        candidates = self._triage(query_embedding, corpus)

        # Compute distance matrices
        query_dists, pairwise_dists = self._compute_distances(
            query_embedding, candidates
        )

        # Greedy selection using Dartboard scoring
        selected_indices = self._greedy_selection(
            query_dists, pairwise_dists, self.config.top_k
        )

        # Get selected chunks and scores
        selected_chunks = [candidates[i] for i in selected_indices]
        scores = [float(query_dists[i]) for i in selected_indices]

        return RetrievalResult(
            query=query,
            chunks=selected_chunks,
            scores=scores,
            metadata={
                "sigma": self.config.sigma,
                "reranker_type": self.config.reranker_type,
                "num_candidates": len(candidates),
            },
        )

    def _triage(self, query_embedding: np.ndarray, corpus: List[Chunk]) -> List[Chunk]:
        """
        Fast KNN-based candidate selection.

        Args:
            query_embedding: Query embedding vector
            corpus: Full corpus of chunks

        Returns:
            Top triage_k candidates by cosine similarity
        """
        # Get all embeddings
        embeddings = np.array([chunk.embedding for chunk in corpus])

        # Compute similarities
        similarities = batch_cosine_similarity(query_embedding, embeddings)

        # Get top-K indices
        top_k = min(self.config.triage_k, len(corpus))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [corpus[int(i)] for i in top_indices]

    def _compute_distances(
        self, query_embedding: np.ndarray, candidates: List[Chunk]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance matrices.

        Args:
            query_embedding: Query embedding
            candidates: Candidate chunks

        Returns:
            (query_dists, pairwise_dists) where:
            - query_dists: (K,) array of query-to-candidate distances
            - pairwise_dists: (K, K) array of candidate-to-candidate distances
        """
        K = len(candidates)
        candidate_embeddings = np.array([c.embedding for c in candidates])

        if self.config.reranker_type == "cosine":
            # Use cosine similarity
            query_sims = batch_cosine_similarity(query_embedding, candidate_embeddings)
            query_dists = 1.0 - query_sims  # Convert to distance

            # Pairwise distances
            pairwise_dists = np.zeros((K, K))
            for i in range(K):
                sims = batch_cosine_similarity(
                    candidate_embeddings[i], candidate_embeddings
                )
                pairwise_dists[i] = 1.0 - sims

        elif self.config.reranker_type == "crossencoder":
            # Use cross-encoder
            if self.cross_encoder is None:
                raise ValueError("Cross-encoder required for crossencoder mode")

            texts = [c.text for c in candidates]
            query_scores = self.cross_encoder.score(query, texts)
            query_dists = -query_scores  # Negate (higher score = lower distance)

            # Pairwise scores
            pairwise_scores = self.cross_encoder.score_matrix(texts)
            pairwise_dists = -pairwise_scores

        else:  # hybrid
            # Cross-encoder for query-passage, cosine for passage-passage
            if self.cross_encoder is None:
                # Fall back to cosine
                return self._compute_distances_cosine_only(query_embedding, candidates)

            texts = [c.text for c in candidates]
            query_scores = self.cross_encoder.score(query, texts)
            query_dists = -query_scores

            # Cosine for pairwise (more efficient)
            pairwise_dists = np.zeros((K, K))
            for i in range(K):
                sims = batch_cosine_similarity(
                    candidate_embeddings[i], candidate_embeddings
                )
                pairwise_dists[i] = 1.0 - sims

        return query_dists, pairwise_dists

    def _compute_distances_cosine_only(
        self, query_embedding: np.ndarray, candidates: List[Chunk]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper for cosine-only distance computation."""
        K = len(candidates)
        candidate_embeddings = np.array([c.embedding for c in candidates])

        query_sims = batch_cosine_similarity(query_embedding, candidate_embeddings)
        query_dists = 1.0 - query_sims

        pairwise_dists = np.zeros((K, K))
        for i in range(K):
            sims = batch_cosine_similarity(
                candidate_embeddings[i], candidate_embeddings
            )
            pairwise_dists[i] = 1.0 - sims

        return query_dists, pairwise_dists

    def _greedy_selection(
        self, query_dists: np.ndarray, pairwise_dists: np.ndarray, k: int
    ) -> List[int]:
        """
        Greedy selection using Dartboard scoring.

        Args:
            query_dists: (K,) array of query-to-candidate distances
            pairwise_dists: (K, K) array of pairwise distances
            k: Number of passages to select

        Returns:
            List of selected indices in order of selection
        """
        K = len(query_dists)
        k = min(k, K)

        # Convert distances to log-probabilities using Gaussian kernel
        sigma = self.config.sigma
        query_log_probs = np.array([-0.5 * (d**2) / (sigma**2) for d in query_dists])

        pairwise_log_probs = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                pairwise_log_probs[i, j] = (
                    -0.5 * (pairwise_dists[i, j] ** 2) / (sigma**2)
                )

        # Initialize with best query match
        selected = []
        available = set(range(K))

        # First selection: highest query probability
        first_idx = int(np.argmax(query_log_probs))
        selected.append(first_idx)
        available.remove(first_idx)

        # Greedy selection for remaining k-1 passages
        for _ in range(k - 1):
            if not available:
                break

            best_score = -np.inf
            best_idx = None

            for candidate_idx in available:
                score = self._compute_dartboard_score(
                    selected, candidate_idx, query_log_probs, pairwise_log_probs
                )

                if score > best_score:
                    best_score = score
                    best_idx = candidate_idx

            if best_idx is not None:
                selected.append(best_idx)
                available.remove(best_idx)

        return selected

    def _compute_dartboard_score(
        self,
        selected_indices: List[int],
        candidate_idx: int,
        query_log_probs: np.ndarray,
        pairwise_log_probs: np.ndarray,
    ) -> float:
        """
        Compute Dartboard score for adding candidate to selected set.

        Implementation of: s(G, q, c) = Σ_t P(t|q) * max(max_{g∈G} N(t|g), N(t|c))

        Args:
            selected_indices: Indices of already selected passages
            candidate_idx: Index of candidate to score
            query_log_probs: Log P(passage|query) for each passage
            pairwise_log_probs: Log P(passage_i|passage_j) matrix

        Returns:
            Dartboard score (higher is better)
        """
        K = len(query_log_probs)
        log_scores = []

        for target_idx in range(K):
            # P(target | query)
            query_prob = query_log_probs[target_idx]

            # max_{g in selected} N(target | g)
            if selected_indices:
                selected_probs = [
                    pairwise_log_probs[target_idx, g] for g in selected_indices
                ]
                max_selected = max(selected_probs)
            else:
                max_selected = -np.inf

            # N(target | candidate)
            candidate_prob = pairwise_log_probs[target_idx, candidate_idx]

            # max(max_selected, candidate)
            max_prob = max(max_selected, candidate_prob)

            # Add to score: query_prob + max_prob (in log space)
            log_scores.append(query_prob + max_prob)

        # Log-sum-exp for numerical stability
        return float(logsumexp_stable(np.array(log_scores)))
