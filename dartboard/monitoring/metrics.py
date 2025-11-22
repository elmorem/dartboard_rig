"""
Prometheus metrics for Dartboard RAG API.

Tracks queries, retrieval/generation latency, errors, and system health.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Metrics Definitions
# ============================================================================

# Query Metrics
query_counter = Counter(
    "rag_queries_total",
    "Total number of RAG queries processed",
    ["status", "tier"],
)

query_errors = Counter(
    "rag_query_errors_total", "Total number of query errors", ["error_type"]
)

# Latency Metrics
retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent retrieving chunks from vector store",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

generation_latency = Histogram(
    "rag_generation_latency_seconds",
    "Time spent generating answers with LLM",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
)

total_query_latency = Histogram(
    "rag_total_query_latency_seconds",
    "Total query processing time (retrieval + generation)",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
)

# Ingestion Metrics
ingestion_counter = Counter(
    "rag_ingestions_total",
    "Total number of document ingestions",
    ["status", "file_type"],
)

ingestion_latency = Histogram(
    "rag_ingestion_latency_seconds",
    "Time spent ingesting documents",
    buckets=[0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

chunks_created = Counter(
    "rag_chunks_created_total", "Total number of chunks created from documents"
)

chunks_stored = Counter(
    "rag_chunks_stored_total", "Total number of chunks stored in vector store"
)

# Vector Store Metrics
vector_store_size = Gauge(
    "rag_vector_store_size", "Current number of chunks in vector store"
)

chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[1, 3, 5, 10, 15, 20, 30, 50],
)

# Authentication Metrics
auth_attempts = Counter(
    "rag_auth_attempts_total",
    "Total authentication attempts",
    ["status", "tier"],
)

# Rate Limiting Metrics
rate_limit_hits = Counter(
    "rag_rate_limit_hits_total",
    "Total number of rate limit violations",
    ["tier"],
)

requests_by_tier = Counter(
    "rag_requests_by_tier_total",
    "Total requests by API key tier",
    ["tier"],
)

# System Info
system_info = Info("rag_system", "RAG system information")


# ============================================================================
# Metric Decorators
# ============================================================================


def track_query_time(tier: str = "unknown"):
    """
    Decorator to track total query time.

    Args:
        tier: API key tier (free, premium, enterprise)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                query_errors.labels(error_type=error_type).inc()
                raise
            finally:
                duration = time.time() - start_time
                total_query_latency.observe(duration)
                query_counter.labels(status=status, tier=tier).inc()

        return wrapper

    return decorator


def track_retrieval_time(func: Callable) -> Callable:
    """Decorator to track retrieval latency."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            retrieval_latency.observe(duration)

    return wrapper


def track_generation_time(func: Callable) -> Callable:
    """Decorator to track generation latency."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            generation_latency.observe(duration)

    return wrapper


def track_ingestion_time(file_type: str = "unknown"):
    """
    Decorator to track ingestion time.

    Args:
        file_type: Type of file being ingested (pdf, md, txt)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                ingestion_latency.observe(duration)
                ingestion_counter.labels(status=status, file_type=file_type).inc()

        return wrapper

    return decorator


# ============================================================================
# Metric Update Functions
# ============================================================================


def record_chunks_retrieved(count: int):
    """Record number of chunks retrieved for a query."""
    chunks_retrieved.observe(count)


def record_chunks_created(count: int):
    """Record number of chunks created from a document."""
    chunks_created.inc(count)


def record_chunks_stored(count: int):
    """Record number of chunks stored in vector store."""
    chunks_stored.inc(count)


def update_vector_store_size(size: int):
    """Update current vector store size."""
    vector_store_size.set(size)


def record_auth_attempt(success: bool, tier: str = "unknown"):
    """Record an authentication attempt."""
    status = "success" if success else "failure"
    auth_attempts.labels(status=status, tier=tier).inc()


def record_rate_limit_hit(tier: str = "unknown"):
    """Record a rate limit violation."""
    rate_limit_hits.labels(tier=tier).inc()


def record_request_by_tier(tier: str):
    """Record a request by API key tier."""
    requests_by_tier.labels(tier=tier).inc()


def set_system_info(
    version: str,
    embedding_model: str,
    llm_model: str,
    vector_store_type: str = "faiss",
):
    """Set system information for monitoring."""
    system_info.info(
        {
            "version": version,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "vector_store_type": vector_store_type,
        }
    )
    logger.info(
        f"System info set: version={version}, "
        f"embedding={embedding_model}, "
        f"llm={llm_model}"
    )


# ============================================================================
# Metrics Export
# ============================================================================


def get_metrics() -> bytes:
    """
    Get Prometheus metrics in text format.

    Returns:
        bytes: Prometheus metrics in text exposition format
    """
    return generate_latest()


def get_metrics_content_type() -> str:
    """
    Get content type for Prometheus metrics.

    Returns:
        str: Content type for metrics endpoint
    """
    return CONTENT_TYPE_LATEST
