"""
Tests for middleware (rate limiting and request logging).

Tests RateLimiter, RateLimitMiddleware, and RequestLoggingMiddleware.
"""

import time
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, HTTPException
from starlette.responses import Response
from dartboard.api.middleware import (
    RateLimiter,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    rate_limiter,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def setup_method(self):
        """Create fresh rate limiter for each test."""
        self.limiter = RateLimiter()

    def test_initial_request_allowed(self):
        """Test first request is always allowed."""
        allowed, info = self.limiter.is_allowed("client1", limit=10)

        assert allowed is True
        assert info["limit"] == 10
        assert info["remaining"] == 9  # One request used
        assert info["used"] == 1

    def test_within_limit_allowed(self):
        """Test requests within limit are allowed."""
        client_id = "client1"
        limit = 5

        for i in range(limit):
            allowed, info = self.limiter.is_allowed(client_id, limit)
            assert allowed is True
            assert info["used"] == i + 1
            assert info["remaining"] == limit - (i + 1)

    def test_exceeding_limit_blocked(self):
        """Test requests exceeding limit are blocked."""
        client_id = "client1"
        limit = 3

        # Use up limit
        for _ in range(limit):
            self.limiter.is_allowed(client_id, limit)

        # Next request should be blocked
        allowed, info = self.limiter.is_allowed(client_id, limit)
        assert allowed is False
        assert info["used"] == limit
        assert info["remaining"] == 0

    def test_different_clients_independent(self):
        """Test different clients have independent rate limits."""
        limit = 5

        # Client 1 uses 3 requests
        for _ in range(3):
            self.limiter.is_allowed("client1", limit)

        # Client 2 should still have full limit
        allowed, info = self.limiter.is_allowed("client2", limit)
        assert allowed is True
        assert info["used"] == 1  # Only their first request

    def test_window_expiration(self):
        """Test requests expire after window."""
        client_id = "client1"
        limit = 2

        # Use up limit
        for _ in range(limit):
            self.limiter.is_allowed(client_id, limit)

        # Should be blocked
        allowed, _ = self.limiter.is_allowed(client_id, limit)
        assert allowed is False

        # Manually expire old requests (simulate 61 seconds passing)
        now = time.time()
        self.limiter.requests[client_id] = [now - 61]

        # Should be allowed again
        allowed, info = self.limiter.is_allowed(client_id, limit)
        assert allowed is True
        assert info["used"] == 1  # Only the new request

    def test_reset_time_calculation(self):
        """Test reset time is calculated correctly."""
        client_id = "client1"
        before_time = time.time()

        allowed, info = self.limiter.is_allowed(client_id, limit=10)

        assert "reset" in info
        reset_time = info["reset"]

        # Reset should be ~60 seconds in future
        expected_reset = before_time + 60
        assert abs(reset_time - expected_reset) < 2  # Allow 2s tolerance

    def test_clear_client(self):
        """Test clearing client rate limit data."""
        client_id = "client1"

        # Make some requests
        for _ in range(3):
            self.limiter.is_allowed(client_id, limit=10)

        assert len(self.limiter.requests[client_id]) == 3

        # Clear client
        self.limiter.clear_client(client_id)
        assert client_id not in self.limiter.requests

    def test_get_stats(self):
        """Test getting rate limiter statistics."""
        # Make requests from different clients
        self.limiter.is_allowed("client1", limit=10)
        self.limiter.is_allowed("client2", limit=10)
        self.limiter.is_allowed("client2", limit=10)

        stats = self.limiter.get_stats()

        assert stats["total_clients"] == 2
        assert stats["active_requests"] == 3

    def test_rate_limit_info_structure(self):
        """Test rate limit info contains all required fields."""
        allowed, info = self.limiter.is_allowed("client1", limit=10)

        assert "limit" in info
        assert "remaining" in info
        assert "reset" in info
        assert "used" in info

        assert isinstance(info["limit"], int)
        assert isinstance(info["remaining"], int)
        assert isinstance(info["reset"], int)
        assert isinstance(info["used"], int)


@pytest.mark.asyncio
class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    def setup_method(self):
        """Create middleware and clear rate limiter before each test."""
        self.middleware = RateLimitMiddleware(app=Mock())
        # Clear global rate limiter
        rate_limiter.requests.clear()

    async def test_health_endpoint_skipped(self):
        """Test rate limiting is skipped for health endpoint."""
        request = Mock(spec=Request)
        request.url.path = "/health"

        call_next = AsyncMock(return_value=Response())

        response = await self.middleware.dispatch(request, call_next)

        assert call_next.called
        assert response.status_code == 200

    async def test_docs_endpoint_skipped(self):
        """Test rate limiting is skipped for docs endpoints."""
        endpoints = ["/", "/docs", "/openapi.json", "/redoc"]

        for path in endpoints:
            request = Mock(spec=Request)
            request.url.path = path
            call_next = AsyncMock(return_value=Response())

            response = await self.middleware.dispatch(request, call_next)
            assert call_next.called

    async def test_authenticated_request_uses_api_key_limit(self):
        """Test authenticated requests use API key rate limit."""
        # Mock request with API key
        request = Mock(spec=Request)
        request.url.path = "/query"
        request.headers.get.return_value = "sk_test_123"
        request.client.host = "127.0.0.1"

        # Mock API key info with premium tier (100 req/min)
        api_key_info = Mock()
        api_key_info.rate_limit = 100
        request.state.api_key_info = api_key_info

        call_next = AsyncMock(return_value=Response())

        response = await self.middleware.dispatch(request, call_next)

        # Should add rate limit headers with premium limit
        assert response.headers["X-RateLimit-Limit"] == "100"

    async def test_unauthenticated_request_uses_ip_limit(self):
        """Test unauthenticated requests use IP-based rate limit."""
        request = Mock(spec=Request)
        request.url.path = "/query"
        request.headers.get.return_value = None  # No API key
        request.client.host = "127.0.0.1"
        request.state = Mock(spec=[])  # No api_key_info

        call_next = AsyncMock(return_value=Response())

        response = await self.middleware.dispatch(request, call_next)

        # Should use default limit of 10
        assert response.headers["X-RateLimit-Limit"] == "10"

    async def test_rate_limit_exceeded(self):
        """Test 429 response when rate limit exceeded."""
        request = Mock(spec=Request)
        request.url.path = "/query"
        request.headers.get.return_value = None
        request.client.host = "127.0.0.1"
        request.state = Mock(spec=[])

        call_next = AsyncMock(return_value=Response())

        # Use up rate limit (default 10)
        for _ in range(10):
            await self.middleware.dispatch(request, call_next)

        # Next request should fail
        with pytest.raises(HTTPException) as exc_info:
            await self.middleware.dispatch(request, call_next)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)

    async def test_rate_limit_headers_on_exceeded(self):
        """Test headers are set correctly when rate limit exceeded."""
        request = Mock(spec=Request)
        request.url.path = "/query"
        request.headers.get.return_value = None
        request.client.host = "127.0.0.1"
        request.state = Mock(spec=[])

        call_next = AsyncMock(return_value=Response())

        # Use up rate limit
        for _ in range(10):
            await self.middleware.dispatch(request, call_next)

        # Next request should fail with headers
        with pytest.raises(HTTPException) as exc_info:
            await self.middleware.dispatch(request, call_next)

        headers = exc_info.value.headers
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert "Retry-After" in headers

        assert headers["X-RateLimit-Remaining"] == "0"

    async def test_rate_limit_headers_added(self):
        """Test rate limit headers are added to successful responses."""
        request = Mock(spec=Request)
        request.url.path = "/query"
        request.headers.get.return_value = None
        request.client.host = "127.0.0.1"
        request.state = Mock(spec=[])

        call_next = AsyncMock(return_value=Response())

        response = await self.middleware.dispatch(request, call_next)

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # First request should have 9 remaining (out of 10)
        assert response.headers["X-RateLimit-Remaining"] == "9"


@pytest.mark.asyncio
class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""

    def setup_method(self):
        """Create middleware before each test."""
        self.middleware = RequestLoggingMiddleware(app=Mock())

    async def test_request_logged(self):
        """Test requests are logged."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/query"
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = "sk_test_123456"

        call_next = AsyncMock(return_value=Response(status_code=200))

        with patch("dartboard.api.middleware.logger") as mock_logger:
            await self.middleware.dispatch(request, call_next)

            # Should log request
            assert mock_logger.info.call_count >= 2  # Request and response

    async def test_response_logged_with_status(self):
        """Test responses are logged with status code."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/health"
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = "none"

        call_next = AsyncMock(return_value=Response(status_code=200))

        with patch("dartboard.api.middleware.logger") as mock_logger:
            await self.middleware.dispatch(request, call_next)

            # Check response log contains status code
            calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("200" in str(call) for call in calls)

    async def test_timing_header_added(self):
        """Test X-Process-Time header is added."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/query"
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = "none"

        call_next = AsyncMock(return_value=Response())

        response = await self.middleware.dispatch(request, call_next)

        assert "X-Process-Time" in response.headers
        assert response.headers["X-Process-Time"].endswith("ms")

    async def test_error_logged(self):
        """Test errors are logged with timing."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/query"
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = "none"

        # Simulate error
        call_next = AsyncMock(side_effect=Exception("Test error"))

        with patch("dartboard.api.middleware.logger") as mock_logger:
            with pytest.raises(Exception):
                await self.middleware.dispatch(request, call_next)

            # Should log error
            mock_logger.error.assert_called_once()

    async def test_api_key_redacted_in_logs(self):
        """Test API keys are redacted in logs."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/query"
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = "sk_test_123456789"

        call_next = AsyncMock(return_value=Response())

        with patch("dartboard.api.middleware.logger") as mock_logger:
            await self.middleware.dispatch(request, call_next)

            # Check that full key is not logged
            calls = [str(call) for call in mock_logger.info.call_args_list]
            assert not any("sk_test_123456789" in str(call) for call in calls)
            # But preview should be logged
            assert any("sk_test_12..." in str(call) for call in calls)

    async def test_missing_client_handled(self):
        """Test requests with no client info are handled."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/health"
        request.client = None  # No client info
        request.headers.get.return_value = "none"

        call_next = AsyncMock(return_value=Response())

        # Should not raise exception
        response = await self.middleware.dispatch(request, call_next)
        assert response.status_code == 200
