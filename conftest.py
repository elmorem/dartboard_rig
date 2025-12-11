"""
Root-level pytest configuration for test files in project root.

Provides shared fixtures for all tests, including configurable embedding models.
"""

import pytest


@pytest.fixture
def embedding_model():
    """
    Provide embedding model based on configuration.

    Can be overridden with EMBEDDING_MODEL environment variable.
    Defaults to configured model (usually all-MiniLM-L6-v2).

    Usage in tests:
        def test_something(embedding_model):
            embeddings = embedding_model.encode("test text")
    """
    from dartboard.embeddings import SentenceTransformerModel
    from dartboard.config import get_embedding_config

    config = get_embedding_config()
    return SentenceTransformerModel(model_name=config.model_name, device=config.device)


@pytest.fixture
def embedding_config():
    """
    Provide embedding configuration for tests.

    Usage in tests:
        def test_config(embedding_config):
            assert embedding_config.embedding_dim in [384, 768]
    """
    from dartboard.config import get_embedding_config

    return get_embedding_config()
