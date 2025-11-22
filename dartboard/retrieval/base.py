"""
Base retriever interface for Dartboard RAG system.

Provides abstract base class for all retrieval methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrievalResult:
    """Results from a retrieval operation."""

    chunks: List[Chunk]
    scores: List[float]
    latency_ms: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "scores": self.scores,
            "latency_ms": self.latency_ms,
            "method": self.method,
            "num_chunks": len(self.chunks),
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval methods.

    All retrieval implementations (Dartboard, BM25, Dense, Hybrid)
    should inherit from this class.
    """

    def __init__(self, vector_store=None):
        """
        Initialize retriever.

        Args:
            vector_store: Vector store for retrieving chunks
        """
        self.vector_store = vector_store

    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> RetrievalResult:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: Search query string
            k: Number of chunks to retrieve
            **kwargs: Additional retriever-specific parameters

        Returns:
            RetrievalResult containing chunks, scores, and metadata
        """
        pass

    def _time_operation(self, func, *args, **kwargs):
        """
        Time an operation and return result with latency.

        Args:
            func: Function to time
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start) * 1000
        return result, latency

    @property
    @abstractmethod
    def name(self) -> str:
        """Return name of retrieval method."""
        pass


class RetrieverNotFittedError(Exception):
    """Raised when retriever is used before being fitted."""

    pass
