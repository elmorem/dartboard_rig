"""
Tests for Prometheus metrics.

Tests metric collection, decorators, and export functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch
from dartboard.monitoring.metrics import (
    # Metrics
    query_counter,
    query_errors,
    retrieval_latency,
    generation_latency,
    total_query_latency,
    ingestion_counter,
    ingestion_latency,
    chunks_created,
    chunks_stored,
    vector_store_size,
    chunks_retrieved,
    auth_attempts,
    rate_limit_hits,
    requests_by_tier,
    system_info,
    # Decorators
    track_query_time,
    track_retrieval_time,
    track_generation_time,
    track_ingestion_time,
    # Functions
    record_chunks_retrieved,
    record_chunks_created,
    record_chunks_stored,
    update_vector_store_size,
    record_auth_attempt,
    record_rate_limit_hit,
    record_request_by_tier,
    set_system_info,
    get_metrics,
    get_metrics_content_type,
)


class TestMetricsExport:
    """Tests for metrics export functionality."""

    def test_get_metrics_returns_bytes(self):
        """Test get_metrics returns bytes."""
        metrics = get_metrics()
        assert isinstance(metrics, bytes)

    def test_get_metrics_contains_help_text(self):
        """Test metrics contain Prometheus help text."""
        metrics = get_metrics().decode("utf-8")
        # Should contain HELP and TYPE declarations
        assert "# HELP" in metrics
        assert "# TYPE" in metrics

    def test_get_metrics_content_type(self):
        """Test metrics content type is correct."""
        content_type = get_metrics_content_type()
        assert isinstance(content_type, str)
        assert "text/plain" in content_type


class TestSystemInfo:
    """Tests for system information metrics."""

    def test_set_system_info(self):
        """Test setting system information."""
        set_system_info(
            version="1.0.0",
            embedding_model="all-MiniLM-L6-v2",
            llm_model="gpt-3.5-turbo",
            vector_store_type="faiss",
        )

        # Get metrics and verify system info is present
        metrics = get_metrics().decode("utf-8")
        assert "rag_system_info" in metrics
        assert "1.0.0" in metrics


class TestCounterMetrics:
    """Tests for counter metrics."""

    def test_query_counter_increment(self):
        """Test query counter increments."""
        # Get initial value
        initial_value = query_counter.labels(
            status="success", tier="free"
        )._value._value

        # Increment
        query_counter.labels(status="success", tier="free").inc()

        # Verify incremented
        new_value = query_counter.labels(status="success", tier="free")._value._value
        assert new_value == initial_value + 1

    def test_query_errors_increment(self):
        """Test query errors counter increments."""
        initial_value = query_errors.labels(error_type="ValueError")._value._value

        query_errors.labels(error_type="ValueError").inc()

        new_value = query_errors.labels(error_type="ValueError")._value._value
        assert new_value == initial_value + 1

    def test_ingestion_counter_with_labels(self):
        """Test ingestion counter with different labels."""
        ingestion_counter.labels(status="success", file_type="pdf").inc()
        ingestion_counter.labels(status="success", file_type="md").inc()
        ingestion_counter.labels(status="error", file_type="pdf").inc()

        # Metrics should track each label combination separately
        metrics = get_metrics().decode("utf-8")
        assert 'file_type="pdf"' in metrics
        assert 'file_type="md"' in metrics


class TestGaugeMetrics:
    """Tests for gauge metrics."""

    def test_update_vector_store_size(self):
        """Test updating vector store size gauge."""
        update_vector_store_size(100)

        # Verify gauge value
        assert vector_store_size._value._value == 100

        # Update again
        update_vector_store_size(200)
        assert vector_store_size._value._value == 200


class TestHistogramMetrics:
    """Tests for histogram metrics."""

    def test_retrieval_latency_observation(self):
        """Test retrieval latency histogram."""
        # Record some observations
        retrieval_latency.observe(0.1)
        retrieval_latency.observe(0.5)
        retrieval_latency.observe(1.0)

        # Verify histogram has observations
        metrics = get_metrics().decode("utf-8")
        assert "rag_retrieval_latency_seconds" in metrics

    def test_generation_latency_observation(self):
        """Test generation latency histogram."""
        generation_latency.observe(2.0)
        generation_latency.observe(5.0)

        metrics = get_metrics().decode("utf-8")
        assert "rag_generation_latency_seconds" in metrics

    def test_chunks_retrieved_histogram(self):
        """Test chunks retrieved histogram."""
        record_chunks_retrieved(5)
        record_chunks_retrieved(10)
        record_chunks_retrieved(15)

        metrics = get_metrics().decode("utf-8")
        assert "rag_chunks_retrieved" in metrics


class TestRecordingFunctions:
    """Tests for metric recording functions."""

    def test_record_chunks_created(self):
        """Test recording chunks created."""
        initial_value = chunks_created._value._value

        record_chunks_created(10)

        new_value = chunks_created._value._value
        assert new_value == initial_value + 10

    def test_record_chunks_stored(self):
        """Test recording chunks stored."""
        initial_value = chunks_stored._value._value

        record_chunks_stored(8)

        new_value = chunks_stored._value._value
        assert new_value == initial_value + 8

    def test_record_auth_attempt_success(self):
        """Test recording successful auth attempt."""
        initial_value = auth_attempts.labels(
            status="success", tier="premium"
        )._value._value

        record_auth_attempt(success=True, tier="premium")

        new_value = auth_attempts.labels(status="success", tier="premium")._value._value
        assert new_value == initial_value + 1

    def test_record_auth_attempt_failure(self):
        """Test recording failed auth attempt."""
        initial_value = auth_attempts.labels(
            status="failure", tier="unknown"
        )._value._value

        record_auth_attempt(success=False, tier="unknown")

        new_value = auth_attempts.labels(status="failure", tier="unknown")._value._value
        assert new_value == initial_value + 1

    def test_record_rate_limit_hit(self):
        """Test recording rate limit hit."""
        initial_value = rate_limit_hits.labels(tier="free")._value._value

        record_rate_limit_hit(tier="free")

        new_value = rate_limit_hits.labels(tier="free")._value._value
        assert new_value == initial_value + 1

    def test_record_request_by_tier(self):
        """Test recording requests by tier."""
        initial_value = requests_by_tier.labels(tier="enterprise")._value._value

        record_request_by_tier(tier="enterprise")

        new_value = requests_by_tier.labels(tier="enterprise")._value._value
        assert new_value == initial_value + 1


@pytest.mark.asyncio
class TestDecorators:
    """Tests for metric decorators."""

    async def test_track_query_time_success(self):
        """Test track_query_time decorator on successful function."""

        @track_query_time(tier="free")
        async def sample_query():
            time.sleep(0.01)  # Simulate work
            return "result"

        initial_count = query_counter.labels(
            status="success", tier="free"
        )._value._value

        result = await sample_query()

        assert result == "result"
        # Counter should increment
        new_count = query_counter.labels(status="success", tier="free")._value._value
        assert new_count == initial_count + 1

    async def test_track_query_time_error(self):
        """Test track_query_time decorator on failed function."""

        @track_query_time(tier="premium")
        async def failing_query():
            raise ValueError("Test error")

        initial_error_count = query_errors.labels(error_type="ValueError")._value._value
        initial_query_count = query_counter.labels(
            status="error", tier="premium"
        )._value._value

        with pytest.raises(ValueError):
            await failing_query()

        # Error counter should increment
        new_error_count = query_errors.labels(error_type="ValueError")._value._value
        assert new_error_count == initial_error_count + 1

        # Query counter (error status) should increment
        new_query_count = query_counter.labels(
            status="error", tier="premium"
        )._value._value
        assert new_query_count == initial_query_count + 1

    def test_track_retrieval_time(self):
        """Test track_retrieval_time decorator."""

        @track_retrieval_time
        def sample_retrieval():
            time.sleep(0.01)
            return ["chunk1", "chunk2"]

        result = sample_retrieval()

        assert result == ["chunk1", "chunk2"]
        # Latency should be recorded (verified by checking metrics exist)
        metrics = get_metrics().decode("utf-8")
        assert "rag_retrieval_latency_seconds" in metrics

    def test_track_generation_time(self):
        """Test track_generation_time decorator."""

        @track_generation_time
        def sample_generation():
            time.sleep(0.01)
            return {"answer": "test"}

        result = sample_generation()

        assert result["answer"] == "test"
        metrics = get_metrics().decode("utf-8")
        assert "rag_generation_latency_seconds" in metrics

    async def test_track_ingestion_time_success(self):
        """Test track_ingestion_time decorator on success."""

        @track_ingestion_time(file_type="pdf")
        async def sample_ingestion():
            time.sleep(0.01)
            return {"chunks": 10}

        initial_count = ingestion_counter.labels(
            status="success", file_type="pdf"
        )._value._value

        result = await sample_ingestion()

        assert result["chunks"] == 10
        new_count = ingestion_counter.labels(
            status="success", file_type="pdf"
        )._value._value
        assert new_count == initial_count + 1

    async def test_track_ingestion_time_error(self):
        """Test track_ingestion_time decorator on error."""

        @track_ingestion_time(file_type="md")
        async def failing_ingestion():
            raise IOError("File not found")

        initial_count = ingestion_counter.labels(
            status="error", file_type="md"
        )._value._value

        with pytest.raises(IOError):
            await failing_ingestion()

        new_count = ingestion_counter.labels(
            status="error", file_type="md"
        )._value._value
        assert new_count == initial_count + 1


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_multiple_metrics_in_export(self):
        """Test that multiple metrics appear in export."""
        # Record various metrics
        query_counter.labels(status="success", tier="free").inc()
        record_chunks_retrieved(5)
        update_vector_store_size(100)
        record_auth_attempt(success=True, tier="premium")

        # Get metrics
        metrics = get_metrics().decode("utf-8")

        # Verify all metrics are present
        assert "rag_queries_total" in metrics
        assert "rag_chunks_retrieved" in metrics
        assert "rag_vector_store_size" in metrics
        assert "rag_auth_attempts_total" in metrics

    def test_histogram_buckets(self):
        """Test histogram metrics include buckets."""
        retrieval_latency.observe(0.5)

        metrics = get_metrics().decode("utf-8")

        # Should include bucket metrics
        assert "rag_retrieval_latency_seconds_bucket" in metrics
        assert "rag_retrieval_latency_seconds_count" in metrics
        assert "rag_retrieval_latency_seconds_sum" in metrics

    def test_labels_preserved_in_export(self):
        """Test that labels are preserved in metrics export."""
        query_counter.labels(status="success", tier="enterprise").inc()
        ingestion_counter.labels(status="error", file_type="txt").inc()

        metrics = get_metrics().decode("utf-8")

        # Check labels are present
        assert 'tier="enterprise"' in metrics
        assert 'file_type="txt"' in metrics
        assert 'status="error"' in metrics
