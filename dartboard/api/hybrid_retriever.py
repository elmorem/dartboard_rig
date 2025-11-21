"""
Hybrid retriever combining vector search with Dartboard refinement.
"""

from __future__ import annotations

from typing import List
import time
from dartboard.core import DartboardRetriever, DartboardConfig
from dartboard.storage.vector_store import VectorStore
from dartboard.embeddings import EmbeddingModel
from dartboard.datasets.models import Chunk, RetrievalResult


class HybridRetriever:
    """
    Two-stage retrieval combining fast vector search with Dartboard diversity.

    Stage 1: Fast KNN search via vector store (retrieve top-N candidates)
    Stage 2: Dartboard algorithm for diversity-aware final selection
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        dartboard_config: DartboardConfig,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector database (FAISS, Pinecone, etc.)
            embedding_model: Model for encoding queries
            dartboard_config: Configuration for Dartboard algorithm
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.dartboard = DartboardRetriever(dartboard_config, embedding_model)

    def retrieve(
        self, query: str, top_k: int = 5, filters: dict = None
    ) -> RetrievalResult:
        """
        Retrieve top-k diverse chunks for query.

        Args:
            query: User query string
            top_k: Number of final chunks to return
            filters: Optional metadata filters

        Returns:
            RetrievalResult with diverse chunks
        """
        # Stage 1: Embed query
        query_embedding = self.embedding_model.encode(query)

        # Stage 2: Fast vector search for candidates
        triage_k = self.dartboard.config.triage_k
        candidates = self.vector_store.search(
            query_embedding, k=triage_k, filters=filters
        )

        if not candidates:
            return RetrievalResult(
                query=query,
                chunks=[],
                scores=[],
                metadata={"stage": "vector_search", "num_candidates": 0},
            )

        # Stage 3: Dartboard refinement for diversity
        result = self.dartboard.retrieve(query, candidates, return_scores=True)

        # Limit to top_k
        result.chunks = result.chunks[:top_k]
        result.scores = result.scores[:top_k]

        # Add metadata
        result.metadata.update(
            {
                "num_candidates": len(candidates),
                "triage_k": triage_k,
                "final_k": len(result.chunks),
            }
        )

        return result

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5, filters: dict = None
    ) -> List[RetrievalResult]:
        """
        Retrieve for multiple queries in batch.

        Args:
            queries: List of query strings
            top_k: Number of chunks per query
            filters: Optional metadata filters

        Returns:
            List of retrieval results
        """
        return [self.retrieve(q, top_k, filters) for q in queries]

    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "vector_store_count": self.vector_store.count(),
            "embedding_dim": self.embedding_model.embedding_dim,
            "dartboard_sigma": self.dartboard.config.sigma,
            "dartboard_triage_k": self.dartboard.config.triage_k,
            "reranker_type": self.dartboard.config.reranker_type,
        }
