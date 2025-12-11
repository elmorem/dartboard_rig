"""Configuration modules for Dartboard."""

from dartboard.config.embeddings import (
    EmbeddingConfig,
    DEFAULT_EMBEDDING_CONFIG,
    get_embedding_config,
)

__all__ = [
    "EmbeddingConfig",
    "DEFAULT_EMBEDDING_CONFIG",
    "get_embedding_config",
]
