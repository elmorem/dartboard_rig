"""
Centralized embedding model configuration for Dartboard.

Provides a single source of truth for embedding model settings
that can be configured via environment variables or programmatically.
"""

import os
from typing import Optional


class EmbeddingConfig:
    """Configuration for embedding models."""

    # Default model settings
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DEVICE = "cpu"

    # Model dimension mapping (common models)
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "multi-qa-mpnet-base-dot-v1": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "intfloat/e5-small-v2": 384,
        "intfloat/e5-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        """
        Initialize embedding configuration.

        Args:
            model_name: Name of the embedding model. If None, uses environment
                       variable EMBEDDING_MODEL or default.
            device: Device to run on ('cpu' or 'cuda'). If None, uses environment
                   variable EMBEDDING_DEVICE or default.
            embedding_dim: Embedding dimension. If None, auto-detected from model
                          or uses environment variable EMBEDDING_DIM.
        """
        # Model name: prioritize param > env var > default
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or os.getenv("EMBEDDING_MODEL_NAME")
            or self.DEFAULT_MODEL
        )

        # Device: prioritize param > env var > default
        self.device = device or os.getenv("EMBEDDING_DEVICE") or self.DEFAULT_DEVICE

        # Embedding dimension: prioritize param > auto-detect > env var > error
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        elif self.model_name in self.MODEL_DIMENSIONS:
            self.embedding_dim = self.MODEL_DIMENSIONS[self.model_name]
        elif os.getenv("EMBEDDING_DIM"):
            self.embedding_dim = int(os.getenv("EMBEDDING_DIM"))
        else:
            # Try to detect from model name
            if "384" in self.model_name or "small" in self.model_name.lower():
                self.embedding_dim = 384
            elif "768" in self.model_name or "base" in self.model_name.lower():
                self.embedding_dim = 768
            else:
                # Default to 384 for unknown models
                self.embedding_dim = 384

    def __repr__(self) -> str:
        return (
            f"EmbeddingConfig(model_name='{self.model_name}', "
            f"device='{self.device}', embedding_dim={self.embedding_dim})"
        )

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create config from environment variables only."""
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL"),
            device=os.getenv("EMBEDDING_DEVICE"),
            embedding_dim=(
                int(os.getenv("EMBEDDING_DIM")) if os.getenv("EMBEDDING_DIM") else None
            ),
        )

    @classmethod
    def get_model_dim(cls, model_name: str) -> int:
        """
        Get embedding dimension for a known model.

        Args:
            model_name: Model name

        Returns:
            Embedding dimension, or 384 as default if unknown
        """
        return cls.MODEL_DIMENSIONS.get(model_name, 384)


# Global default config instance (can be overridden)
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


def get_embedding_config(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    embedding_dim: Optional[int] = None,
) -> EmbeddingConfig:
    """
    Get embedding configuration.

    If no parameters provided, returns the global default config.
    Otherwise, creates a new config with the specified parameters.

    Args:
        model_name: Optional model name override
        device: Optional device override
        embedding_dim: Optional dimension override

    Returns:
        EmbeddingConfig instance
    """
    if model_name is None and device is None and embedding_dim is None:
        return DEFAULT_EMBEDDING_CONFIG
    return EmbeddingConfig(model_name, device, embedding_dim)
