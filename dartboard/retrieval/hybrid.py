"""
Hybrid retriever implementation.

Combines BM25 (sparse) and Dense (vector) retrieval using
Reciprocal Rank Fusion (RRF) for ranking.
"""

import logging
import time
from typing import List, Dict, Tuple
from collections import defaultdict

from dartboard.retrieval.base import BaseRetriever, RetrievalResult, Chunk
from dartboard.retrieval.bm25 import BM25Retriever
from dartboard.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining BM25 and Dense retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge results from
    sparse (BM25) and dense (vector) retrieval methods.

    RRF Formula:
    score(doc) = sum over methods: 1 / (k + rank(doc))

    Advantages:
    - Combines lexical and semantic matching
    - Better than either method alone
    - Robust across query types
    - No parameter tuning needed (RRF is parameter-free)

    Disadvantages:
    - Slower (runs both retrievers)
    - More complex implementation
    """

    def __init__(
        self,
        vector_store=None,
        bm25_retriever: BM25Retriever = None,
        dense_retriever: DenseRetriever = None,
        k_rrf: int = 60,
        weight_bm25: float = 0.5,
        weight_dense: float = 0.5,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector store (passed to retrievers)
            bm25_retriever: Pre-initialized BM25 retriever (optional)
            dense_retriever: Pre-initialized Dense retriever (optional)
            k_rrf: RRF constant (default: 60, recommended in literature)
            weight_bm25: Weight for BM25 scores (default: 0.5)
            weight_dense: Weight for dense scores (default: 0.5)
        """
        super().__init__(vector_store)

        # Initialize component retrievers
        self.bm25 = bm25_retriever or BM25Retriever(vector_store)
        self.dense = dense_retriever or DenseRetriever(vector_store)

        self.k_rrf = k_rrf
        self.weight_bm25 = weight_bm25
        self.weight_dense = weight_dense

        logger.info(
            f"Initialized hybrid retriever (k_rrf={k_rrf}, "
            f"bm25_weight={weight_bm25}, dense_weight={weight_dense})"
        )

    def fit(self, chunks: List[Chunk] = None):
        """
        Fit both BM25 and Dense retrievers.

        Args:
            chunks: Chunks to index (if None, uses vector_store)
        """
        logger.info("Fitting hybrid retriever components")

        # Fit BM25
        if not self.bm25._is_fitted:
            self.bm25.fit(chunks)

        logger.info("Hybrid retriever ready")

    def retrieve(
        self, query: str, k: int = 5, retrieve_k_multiplier: int = 3, **kwargs
    ) -> RetrievalResult:
        """
        Retrieve top-k chunks using hybrid retrieval with RRF.

        Args:
            query: Search query string
            k: Number of final chunks to return
            retrieve_k_multiplier: Retrieve k*multiplier from each method
                                   before fusion (default: 3)
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with fused rankings
        """
        start = time.time()

        # Retrieve from both methods (get more candidates for better fusion)
        retrieve_k = k * retrieve_k_multiplier

        logger.debug(f"Retrieving {retrieve_k} candidates from each method")

        # BM25 retrieval
        bm25_results = self.bm25.retrieve(query, k=retrieve_k)

        # Dense retrieval
        dense_results = self.dense.retrieve(query, k=retrieve_k)

        # Apply Reciprocal Rank Fusion
        fused_chunks, fused_scores = self._reciprocal_rank_fusion(
            bm25_results=bm25_results, dense_results=dense_results, k=k
        )

        latency = (time.time() - start) * 1000

        logger.debug(
            f"Hybrid retrieval: {k} chunks in {latency:.2f}ms, "
            f"top score: {fused_scores[0]:.4f}"
        )

        return RetrievalResult(
            chunks=fused_chunks,
            scores=fused_scores,
            latency_ms=latency,
            method="hybrid",
            metadata={
                "k_rrf": self.k_rrf,
                "weight_bm25": self.weight_bm25,
                "weight_dense": self.weight_dense,
                "bm25_latency_ms": bm25_results.latency_ms,
                "dense_latency_ms": dense_results.latency_ms,
                "candidates_retrieved": retrieve_k,
            },
        )

    def _reciprocal_rank_fusion(
        self, bm25_results: RetrievalResult, dense_results: RetrievalResult, k: int
    ) -> Tuple[List[Chunk], List[float]]:
        """
        Merge BM25 and Dense results using Reciprocal Rank Fusion.

        RRF score for document d:
        score(d) = sum over all methods m: weight_m * (1 / (k + rank_m(d)))

        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from Dense retriever
            k: Number of top results to return

        Returns:
            Tuple of (merged_chunks, merged_scores)
        """
        # Map chunk ID to (chunk, rrf_score, source_info)
        chunk_scores = {}

        # Process BM25 results
        for rank, chunk in enumerate(bm25_results.chunks, 1):
            rrf_score = self.weight_bm25 / (self.k_rrf + rank)
            if chunk.id in chunk_scores:
                chunk_scores[chunk.id]["score"] += rrf_score
                chunk_scores[chunk.id]["sources"].append("bm25")
            else:
                chunk_scores[chunk.id] = {
                    "chunk": chunk,
                    "score": rrf_score,
                    "sources": ["bm25"],
                    "bm25_rank": rank,
                }

        # Process Dense results
        for rank, chunk in enumerate(dense_results.chunks, 1):
            rrf_score = self.weight_dense / (self.k_rrf + rank)
            if chunk.id in chunk_scores:
                chunk_scores[chunk.id]["score"] += rrf_score
                chunk_scores[chunk.id]["sources"].append("dense")
                chunk_scores[chunk.id]["dense_rank"] = rank
            else:
                chunk_scores[chunk.id] = {
                    "chunk": chunk,
                    "score": rrf_score,
                    "sources": ["dense"],
                    "dense_rank": rank,
                }

        # Sort by RRF score
        sorted_items = sorted(
            chunk_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        # Get top-k
        top_k_items = sorted_items[:k]

        # Extract chunks and scores
        merged_chunks = []
        merged_scores = []

        for chunk_id, info in top_k_items:
            chunk = info["chunk"]
            chunk.score = info["score"]

            # Add fusion metadata to chunk
            chunk.metadata["rrf_score"] = info["score"]
            chunk.metadata["sources"] = info["sources"]
            chunk.metadata["in_both"] = len(info["sources"]) == 2

            if "bm25_rank" in info:
                chunk.metadata["bm25_rank"] = info["bm25_rank"]
            if "dense_rank" in info:
                chunk.metadata["dense_rank"] = info["dense_rank"]

            merged_chunks.append(chunk)
            merged_scores.append(info["score"])

        return merged_chunks, merged_scores

    @property
    def name(self) -> str:
        """Return retriever name."""
        return "hybrid"

    def get_fusion_analysis(
        self, bm25_results: RetrievalResult, dense_results: RetrievalResult
    ) -> Dict:
        """
        Analyze overlap and disagreement between BM25 and Dense results.

        Args:
            bm25_results: Results from BM25
            dense_results: Results from Dense

        Returns:
            Analysis dict with overlap stats
        """
        bm25_ids = {chunk.id for chunk in bm25_results.chunks}
        dense_ids = {chunk.id for chunk in dense_results.chunks}

        overlap = bm25_ids & dense_ids
        bm25_only = bm25_ids - dense_ids
        dense_only = dense_ids - bm25_ids

        return {
            "overlap_count": len(overlap),
            "overlap_percentage": len(overlap) / len(bm25_ids) * 100,
            "bm25_only_count": len(bm25_only),
            "dense_only_count": len(dense_only),
            "total_unique": len(bm25_ids | dense_ids),
        }
