"""
Dependency injection for FastAPI routes.

Provides singleton instances of:
- Embedding model
- Vector store
- Ingestion pipeline
- RAG generator
- Hybrid retriever
"""

import os
import logging
from typing import Optional
from functools import lru_cache

from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore
from dartboard.ingestion.pipeline import create_pipeline
from dartboard.ingestion.loaders import MarkdownLoader
from dartboard.generation.generator import create_generator
from dartboard.api.hybrid_retriever import HybridRetriever
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.config import get_embedding_config

logger = logging.getLogger(__name__)


# Get embedding configuration from centralized config
EMBEDDING_CONFIG = get_embedding_config()

# Other configuration from environment variables
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@lru_cache()
def get_embedding_model() -> SentenceTransformerModel:
    """
    Get singleton embedding model.

    Returns:
        SentenceTransformerModel instance
    """
    logger.info(
        f"Initializing embedding model: {EMBEDDING_CONFIG.model_name} "
        f"(dim={EMBEDDING_CONFIG.embedding_dim}, device={EMBEDDING_CONFIG.device})"
    )
    return SentenceTransformerModel(
        model_name=EMBEDDING_CONFIG.model_name, device=EMBEDDING_CONFIG.device
    )


@lru_cache()
def get_vector_store() -> FAISSStore:
    """
    Get singleton vector store.

    Returns:
        FAISSStore instance
    """
    logger.info(
        f"Initializing vector store at {VECTOR_STORE_PATH} "
        f"(dim={EMBEDDING_CONFIG.embedding_dim})"
    )
    return FAISSStore(embedding_dim=EMBEDDING_CONFIG.embedding_dim)


def get_ingestion_pipeline():
    """
    Get ingestion pipeline instance.

    Returns:
        IngestionPipeline configured with loader, chunker, embedder, store
    """
    embedding_model = get_embedding_model()
    vector_store = get_vector_store()
    loader = MarkdownLoader()  # Default to Markdown

    pipeline = create_pipeline(
        loader=loader,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=512,
        overlap=50,
        batch_size=32,
    )

    return pipeline


@lru_cache()
def get_rag_generator():
    """
    Get RAG generator instance.

    Returns:
        RAGGenerator instance
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable required for RAG generation"
        )

    logger.info(f"Initializing RAG generator: {LLM_PROVIDER}/{LLM_MODEL}")

    return create_generator(
        provider=LLM_PROVIDER,
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=500,
    )


def get_hybrid_retriever():
    """
    Get hybrid retriever instance.

    Returns:
        HybridRetriever combining vector search and Dartboard
    """
    embedding_model = get_embedding_model()
    vector_store = get_vector_store()

    # Create Dartboard retriever
    dartboard_config = DartboardConfig(sigma=1.0, top_k=5, triage_k=100)
    dartboard_retriever = DartboardRetriever(dartboard_config, embedding_model)

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        dartboard_retriever=dartboard_retriever,
        embedding_model=embedding_model,
    )

    return hybrid_retriever


def get_config():
    """
    Get API configuration.

    Returns:
        Dictionary of configuration values
    """
    return {
        "embedding_model": EMBEDDING_CONFIG.model_name,
        "embedding_dim": EMBEDDING_CONFIG.embedding_dim,
        "embedding_device": EMBEDDING_CONFIG.device,
        "vector_store_path": VECTOR_STORE_PATH,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
    }
