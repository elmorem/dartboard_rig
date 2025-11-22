"""
Cross-encoder reranker implementation.

Reranks retrieved chunks using a cross-encoder model for
more accurate relevance scoring.
"""

import logging
import time
from typing import List
import numpy as np
from sentence_transformers import CrossEncoder

from dartboard.retrieval.base import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval results.

    Uses a cross-encoder model to score query-document pairs
    and rerank initial retrieval results for better precision.

    Two-stage retrieval pipeline:
    1. Fast retrieval (BM25/Dense/Hybrid) - get candidates
    2. Slow reranking (Cross-encoder) - rerank top candidates

    Advantages:
    - More accurate than bi-encoders
    - Better relevance scores
    - Can capture query-document interactions

    Disadvantages:
    - Slow (must score each query-doc pair)
    - Cannot be used for first-stage retrieval
    - Requires pretrained cross-encoder model
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Hugging Face model name (default: MS MARCO MiniLM)
            batch_size: Batch size for scoring (default: 32)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded cross-encoder: {model_name}")

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = None,
    ) -> RetrievalResult:
        """
        Rerank chunks using cross-encoder scores.

        Args:
            query: Search query
            chunks: Chunks to rerank
            top_k: Number of top chunks to return (default: all)

        Returns:
            RetrievalResult with reranked chunks
        """
        if not chunks:
            return RetrievalResult(
                chunks=[],
                scores=[],
                latency_ms=0.0,
                method="reranker",
            )

        start = time.time()

        # Create query-document pairs
        pairs = [[query, chunk.text] for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        # Convert to list of floats
        scores = [float(score) for score in scores]

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Rerank chunks
        reranked_chunks = [chunks[i] for i in sorted_indices]
        reranked_scores = [scores[i] for i in sorted_indices]

        # Update chunk scores
        for chunk, score in zip(reranked_chunks, reranked_scores):
            chunk.score = score
            chunk.metadata["reranker_score"] = score

        # Get top-k if specified
        if top_k is not None:
            reranked_chunks = reranked_chunks[:top_k]
            reranked_scores = reranked_scores[:top_k]

        latency = (time.time() - start) * 1000

        logger.debug(
            f"Reranked {len(chunks)} chunks in {latency:.2f}ms, "
            f"top score: {reranked_scores[0]:.4f}"
        )

        return RetrievalResult(
            chunks=reranked_chunks,
            scores=reranked_scores,
            latency_ms=latency,
            method="reranker",
            metadata={
                "model": self.model_name,
                "input_chunks": len(chunks),
                "output_chunks": len(reranked_chunks),
            },
        )

    def score_pair(self, query: str, text: str) -> float:
        """
        Score a single query-text pair.

        Args:
            query: Query string
            text: Document text

        Returns:
            Relevance score
        """
        score = self.model.predict([[query, text]])[0]
        return float(score)
