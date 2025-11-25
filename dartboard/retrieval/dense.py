"""
Dense retriever implementation.

Implements dense retrieval using vector similarity (cosine similarity)
with sentence transformer embeddings.
"""

import logging
import time
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from dartboard.retrieval.base import BaseRetriever, RetrievalResult, Chunk

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using vector similarity.

    Uses sentence transformer embeddings and cosine similarity
    for semantic retrieval.

    Advantages:
    - Captures semantic meaning
    - Good for paraphrased/synonym queries
    - Works with multilingual queries (if model supports)
    - Handles vocabulary mismatch well

    Disadvantages:
    - Slower than BM25 (embedding generation)
    - Requires embedding model
    - Less explainable
    - May miss exact keyword matches
    """

    def __init__(
        self,
        vector_store=None,
        embedding_model: Optional[SentenceTransformer] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_metric: str = "cosine",
    ):
        """
        Initialize dense retriever.

        Args:
            vector_store: Vector store with pre-computed embeddings
            embedding_model: Pre-loaded embedding model (optional)
            model_name: Name of sentence transformer model to load
            similarity_metric: Similarity metric ('cosine', 'dot', or 'euclidean')
        """
        super().__init__(vector_store)
        self.embedding_model = embedding_model or SentenceTransformer(model_name)
        self.model_name = model_name
        self.similarity_metric = similarity_metric
        logger.info(f"Initialized dense retriever with {model_name}")

    def retrieve(self, query: str, k: int = 5, **kwargs) -> RetrievalResult:
        """
        Retrieve top-k chunks using dense vector similarity.

        Args:
            query: Search query string
            k: Number of chunks to retrieve
            **kwargs: Additional parameters (e.g., 'score_threshold')

        Returns:
            RetrievalResult with ranked chunks and similarity scores
        """
        start = time.time()

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        )

        # Use vector store for similarity search
        if self.vector_store is not None:
            results = self.vector_store.similarity_search(
                query_embedding=query_embedding, k=k, metric=self.similarity_metric
            )
            chunks = results["chunks"]
            scores = results["scores"]
        else:
            raise ValueError("Vector store required for dense retrieval")

        # Update chunk scores
        for chunk, score in zip(chunks, scores):
            chunk.score = score

        latency = (time.time() - start) * 1000

        logger.debug(
            f"Dense retrieval: {k} chunks in {latency:.2f}ms, "
            f"top score: {scores[0]:.4f}"
        )

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            latency_ms=latency,
            method="dense",
            metadata={
                "model": self.model_name,
                "similarity_metric": self.similarity_metric,
                "embedding_dim": len(query_embedding),
            },
        )

    @property
    def name(self) -> str:
        """Return retriever name."""
        return "dense"

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings
        """
        embeddings = self.embedding_model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )
        return embeddings
