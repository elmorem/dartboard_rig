"""
BM25 retriever implementation.

Implements sparse retrieval using the BM25 (Best Matching 25) algorithm,
which uses term frequency and inverse document frequency for ranking.
"""

import logging
import time
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from dartboard.retrieval.base import (
    BaseRetriever,
    RetrievalResult,
    Chunk,
    RetrieverNotFittedError,
)

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """
    BM25 (Best Matching 25) retriever.

    Uses term frequency and inverse document frequency for ranking documents.
    Particularly effective for keyword-based queries and exact match retrieval.

    Advantages:
    - Fast retrieval
    - Good for keyword/lexical matching
    - No need for embeddings
    - Explainable (term-based scores)

    Disadvantages:
    - Sensitive to vocabulary mismatch
    - Poor with semantic/paraphrased queries
    - Requires tokenization strategy
    """

    def __init__(
        self,
        vector_store=None,
        tokenizer=None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 retriever.

        Args:
            vector_store: Vector store containing chunks
            tokenizer: Custom tokenizer function (default: simple whitespace split)
            k1: BM25 k1 parameter (term frequency saturation, default: 1.5)
            b: BM25 b parameter (length normalization, default: 0.75)
        """
        super().__init__(vector_store)
        self.tokenizer = tokenizer or self._default_tokenizer
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = []
        self._is_fitted = False

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Default tokenizer: lowercase and split on whitespace."""
        return text.lower().split()

    def fit(self, chunks: Optional[List[Chunk]] = None):
        """
        Fit BM25 model on chunk corpus.

        Args:
            chunks: List of chunks to index. If None, uses vector_store.

        Raises:
            ValueError: If no chunks provided and no vector_store
        """
        if chunks is None:
            if self.vector_store is None:
                raise ValueError("Either chunks or vector_store must be provided")
            chunks = self.vector_store.get_all_chunks()

        if not chunks:
            raise ValueError("No chunks to fit BM25 model")

        self.chunks = chunks
        logger.info(f"Fitting BM25 on {len(chunks)} chunks")

        # Tokenize corpus
        tokenized_corpus = [self.tokenizer(chunk.text) for chunk in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        self._is_fitted = True
        logger.info("BM25 indexing complete")

    def retrieve(self, query: str, k: int = 5, **kwargs) -> RetrievalResult:
        """
        Retrieve top-k chunks using BM25.

        Args:
            query: Search query string
            k: Number of chunks to retrieve
            **kwargs: Additional parameters (ignored)

        Returns:
            RetrievalResult with ranked chunks and scores

        Raises:
            RetrieverNotFittedError: If fit() hasn't been called
        """
        if not self._is_fitted:
            raise RetrieverNotFittedError("BM25 retriever must be fitted before use")

        start = time.time()

        # Tokenize query
        tokenized_query = self.tokenizer(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        # Get chunks and scores
        top_chunks = [self.chunks[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        # Update chunk scores
        for chunk, score in zip(top_chunks, top_scores):
            chunk.score = score

        latency = (time.time() - start) * 1000

        logger.debug(
            f"BM25 retrieval: {k} chunks in {latency:.2f}ms, "
            f"top score: {top_scores[0]:.4f}"
        )

        return RetrievalResult(
            chunks=top_chunks,
            scores=top_scores,
            latency_ms=latency,
            method="bm25",
            metadata={
                "k1": self.k1,
                "b": self.b,
                "query_length": len(tokenized_query),
                "corpus_size": len(self.chunks),
            },
        )

    @property
    def name(self) -> str:
        """Return retriever name."""
        return "bm25"

    def get_term_scores(self, query: str, doc_idx: int) -> dict:
        """
        Get per-term BM25 scores for a document (for analysis).

        Args:
            query: Query string
            doc_idx: Document index

        Returns:
            Dict mapping terms to their BM25 contribution scores
        """
        if not self._is_fitted:
            raise RetrieverNotFittedError("BM25 retriever must be fitted before use")

        tokenized_query = self.tokenizer(query)
        doc_scores = {}

        for term in set(tokenized_query):
            # Get term score from BM25
            term_score = self.bm25.get_scores([term])[doc_idx]
            doc_scores[term] = float(term_score)

        return doc_scores
