from __future__ import annotations

"""
Embedding model wrappers for Dartboard RAG.

Supports sentence transformers and cross-encoder models.
"""

from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) to dense vectors.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding

        Returns:
            Embeddings array of shape (n, dim) or (dim,)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        pass


class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for sentence-transformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize sentence transformer model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu' or 'cuda')
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        embeddings = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True
        )
        # Return single embedding as 1D array, multiple as 2D
        if is_single and embeddings.ndim > 1:
            return embeddings[0]
        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class CrossEncoder:
    """Wrapper for cross-encoder reranking models."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = "cpu",
    ):
        """
        Initialize cross-encoder model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
        """
        from sentence_transformers import CrossEncoder as CE

        self.model = CE(model_name, device=device)
        self.model_name = model_name
        self.device = device

    def score(
        self, query: str, passages: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Score query-passage pairs.

        Args:
            query: Query text
            passages: List of passage texts
            batch_size: Batch size for inference

        Returns:
            Array of relevance scores (higher = more relevant)
        """
        pairs = [(query, passage) for passage in passages]
        return self.model.predict(pairs, batch_size=batch_size)

    def score_matrix(self, passages: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Compute pairwise passage similarity scores.

        Args:
            passages: List of passage texts
            batch_size: Batch size for inference

        Returns:
            (N, N) symmetric matrix of scores
        """
        n = len(passages)
        scores = np.zeros((n, n))

        # Compute upper triangle (symmetric matrix)
        pairs = []
        indices = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((passages[i], passages[j]))
                indices.append((i, j))

        # Get scores in batches
        pair_scores = self.model.predict(pairs, batch_size=batch_size)

        # Fill matrix
        for (i, j), score in zip(indices, pair_scores):
            scores[i, j] = score
            scores[j, i] = score  # Symmetric

        return scores
