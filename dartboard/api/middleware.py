"""
Middleware for Dartboard RAG API.

Provides rate limiting and request logging.
"""

import logging
import time
from typing import Dict, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter.

    Tracks requests per client and enforces limits based on API key tier.
    """

    def __init__(self):
        # Store: {client_id: [(timestamp, count)]}
        self.requests: Dict[str, list] = defaultdict(list)
        self.window_seconds = 60  # 1 minute window

    def is_allowed(self, client_id: str, limit: int) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limit.

        Args:
            client_id: Client identifier (API key or IP)
            limit: Maximum requests per minute

        Returns:
            Tuple of (allowed: bool, info: dict with rate limit info)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > window_start
        ]

        # Count requests in current window
        request_count = len(self.requests[client_id])

        # Calculate reset time
        if self.requests[client_id]:
            oldest_request = min(self.requests[client_id])
            reset_time = oldest_request + self.window_seconds
        else:
            reset_time = now + self.window_seconds

        info = {
            "limit": limit,
            "remaining": max(0, limit - request_count),
            "reset": int(reset_time),
            "used": request_count,
        }

        # Check if allowed
        if request_count >= limit:
            logger.warning(
                f"Rate limit exceeded for {client_id}: {request_count}/{limit}"
            )
            return False, info

        # Record this request
        self.requests[client_id].append(now)

        return True, info

    def clear_client(self, client_id: str):
        """Clear rate limit data for a client."""
        if client_id in self.requests:
            del self.requests[client_id]

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "total_clients": len(self.requests),
            "active_requests": sum(len(reqs) for reqs in self.requests.values()),
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.

    Applies rate limits based on API key tier or IP address.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""

        # Skip rate limiting for health/docs/metrics endpoints
        if request.url.path in [
            "/health",
            "/",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/metrics",
        ]:
            return await call_next(request)

        # Get client identifier (API key or IP)
        api_key = request.headers.get("x-api-key")
        client_id = api_key if api_key else request.client.host

        # Determine rate limit
        # Default to 10 req/min for non-authenticated requests
        rate_limit = 10

        # Get tier-specific limit from request state (set by auth middleware)
        if hasattr(request.state, "api_key_info"):
            rate_limit = request.state.api_key_info.rate_limit

        # Check rate limit
        allowed, info = rate_limiter.is_allowed(client_id, rate_limit)

        if not allowed:
            # Calculate retry-after time
            retry_after = info["reset"] - int(time.time())

            logger.warning(
                f"Rate limit exceeded for {client_id}: "
                f"{info['used']}/{info['limit']} requests"
            )

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {info['limit']}/minute",
                    "retry_after": retry_after,
                    "rate_limit": info,
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(retry_after),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests.

    Logs request method, path, status code, and duration.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request with logging."""
        start_time = time.time()

        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("x-api-key", "none")
        api_key_preview = api_key[:10] + "..." if api_key != "none" else "none"

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} (key: {api_key_preview})"
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"for {request.method} {request.url.path} "
                f"({duration_ms:.2f}ms)"
            )

            # Add timing header
            response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            # Log errors
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error: {request.method} {request.url.path} "
                f"({duration_ms:.2f}ms): {str(e)}"
            )
            raise
