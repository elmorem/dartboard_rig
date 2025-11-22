"""
Tests for authentication and authorization.

Tests API key validation, tier management, and security.
"""

import pytest
from fastapi import HTTPException
from dartboard.api.auth import (
    verify_api_key,
    verify_api_key_optional,
    add_api_key,
    revoke_api_key,
    get_all_keys,
    APIKeyInfo,
    VALID_API_KEYS,
    RATE_LIMITS,
)


class TestAPIKeyInfo:
    """Tests for APIKeyInfo class."""

    def test_api_key_info_initialization(self):
        """Test APIKeyInfo initialization with valid metadata."""
        metadata = {
            "name": "Test User",
            "tier": "premium",
            "created_at": "2025-01-01T00:00:00Z",
            "active": True,
        }
        key_info = APIKeyInfo("test_key", metadata)

        assert key_info.key == "test_key"
        assert key_info.name == "Test User"
        assert key_info.tier == "premium"
        assert key_info.created_at == "2025-01-01T00:00:00Z"
        assert key_info.active is True

    def test_api_key_info_defaults(self):
        """Test APIKeyInfo with missing metadata fields."""
        metadata = {}
        key_info = APIKeyInfo("test_key", metadata)

        assert key_info.name == "Unknown"
        assert key_info.tier == "free"
        assert key_info.active is True

    def test_rate_limit_by_tier(self):
        """Test rate limit retrieval for different tiers."""
        tiers = {
            "free": 10,
            "premium": 100,
            "enterprise": 1000,
        }

        for tier, expected_limit in tiers.items():
            metadata = {"tier": tier}
            key_info = APIKeyInfo("test_key", metadata)
            assert key_info.rate_limit == expected_limit

    def test_rate_limit_unknown_tier(self):
        """Test rate limit defaults to free tier for unknown tiers."""
        metadata = {"tier": "unknown_tier"}
        key_info = APIKeyInfo("test_key", metadata)
        assert key_info.rate_limit == 10  # Default to free tier

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = {
            "name": "Test User",
            "tier": "premium",
            "created_at": "2025-01-01T00:00:00Z",
            "active": True,
        }
        key_info = APIKeyInfo("test_key", metadata)
        result = key_info.to_dict()

        assert result["name"] == "Test User"
        assert result["tier"] == "premium"
        assert result["created_at"] == "2025-01-01T00:00:00Z"
        assert result["active"] is True
        assert result["rate_limit"] == 100


@pytest.mark.asyncio
class TestVerifyAPIKey:
    """Tests for API key verification."""

    async def test_valid_api_key(self):
        """Test verification with valid API key."""
        key_info = await verify_api_key(x_api_key="sk_test_123456")

        assert isinstance(key_info, APIKeyInfo)
        assert key_info.name == "Test User"
        assert key_info.tier == "free"
        assert key_info.active is True

    async def test_missing_api_key(self):
        """Test verification fails when API key is missing."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key=None)

        assert exc_info.value.status_code == 401
        assert "API key required" in exc_info.value.detail

    async def test_invalid_api_key(self):
        """Test verification fails with invalid API key."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="invalid_key")

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    async def test_inactive_api_key(self):
        """Test verification fails with inactive API key."""
        # Add inactive key
        add_api_key("sk_test_inactive", "Inactive User", "free")
        revoke_api_key("sk_test_inactive")

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="sk_test_inactive")

        assert exc_info.value.status_code == 401
        assert "deactivated" in exc_info.value.detail

        # Cleanup
        del VALID_API_KEYS["sk_test_inactive"]

    async def test_premium_api_key(self):
        """Test verification with premium API key."""
        key_info = await verify_api_key(x_api_key="sk_prod_789012")

        assert key_info.name == "Production User"
        assert key_info.tier == "premium"
        assert key_info.rate_limit == 100


@pytest.mark.asyncio
class TestVerifyAPIKeyOptional:
    """Tests for optional API key verification."""

    async def test_valid_api_key_optional(self):
        """Test optional verification with valid key."""
        key_info = await verify_api_key_optional(x_api_key="sk_test_123456")

        assert isinstance(key_info, APIKeyInfo)
        assert key_info.name == "Test User"

    async def test_missing_api_key_optional(self):
        """Test optional verification returns None when key missing."""
        key_info = await verify_api_key_optional(x_api_key=None)
        assert key_info is None

    async def test_invalid_api_key_optional(self):
        """Test optional verification returns None for invalid key."""
        key_info = await verify_api_key_optional(x_api_key="invalid_key")
        assert key_info is None

    async def test_inactive_api_key_optional(self):
        """Test optional verification returns None for inactive key."""
        # Add inactive key
        add_api_key("sk_test_inactive2", "Inactive User", "free")
        revoke_api_key("sk_test_inactive2")

        key_info = await verify_api_key_optional(x_api_key="sk_test_inactive2")
        assert key_info is None

        # Cleanup
        del VALID_API_KEYS["sk_test_inactive2"]


class TestAPIKeyManagement:
    """Tests for API key management functions."""

    def test_add_api_key(self):
        """Test adding a new API key."""
        key = "sk_test_new_key"

        # Clean up if exists
        if key in VALID_API_KEYS:
            del VALID_API_KEYS[key]

        result = add_api_key(key, "New User", "premium")
        assert result is True
        assert key in VALID_API_KEYS
        assert VALID_API_KEYS[key]["name"] == "New User"
        assert VALID_API_KEYS[key]["tier"] == "premium"
        assert VALID_API_KEYS[key]["active"] is True

        # Cleanup
        del VALID_API_KEYS[key]

    def test_add_duplicate_api_key(self):
        """Test adding duplicate API key fails."""
        key = "sk_test_duplicate"

        # Add first time
        add_api_key(key, "User 1", "free")

        # Try to add again
        result = add_api_key(key, "User 2", "premium")
        assert result is False

        # Original key should be unchanged
        assert VALID_API_KEYS[key]["name"] == "User 1"
        assert VALID_API_KEYS[key]["tier"] == "free"

        # Cleanup
        del VALID_API_KEYS[key]

    def test_revoke_api_key(self):
        """Test revoking an API key."""
        key = "sk_test_revoke"

        # Add key
        add_api_key(key, "User", "free")
        assert VALID_API_KEYS[key]["active"] is True

        # Revoke key
        result = revoke_api_key(key)
        assert result is True
        assert VALID_API_KEYS[key]["active"] is False

        # Cleanup
        del VALID_API_KEYS[key]

    def test_revoke_nonexistent_key(self):
        """Test revoking non-existent key fails."""
        result = revoke_api_key("sk_test_nonexistent")
        assert result is False

    def test_get_all_keys(self):
        """Test retrieving all API keys."""
        keys = get_all_keys()

        assert isinstance(keys, dict)
        assert len(keys) >= 2  # At least test and prod keys

        # Check that keys are redacted
        for key_preview, metadata in keys.items():
            assert key_preview.endswith("...")
            assert len(key_preview) == 13  # "sk_test_12..." = 13 chars
            assert "name" in metadata
            assert "tier" in metadata
            assert "key_prefix" in metadata


class TestRateLimits:
    """Tests for rate limit configuration."""

    def test_rate_limits_defined(self):
        """Test that rate limits are defined for all tiers."""
        assert "free" in RATE_LIMITS
        assert "premium" in RATE_LIMITS
        assert "enterprise" in RATE_LIMITS

    def test_rate_limits_values(self):
        """Test rate limit values are reasonable."""
        assert RATE_LIMITS["free"] == 10
        assert RATE_LIMITS["premium"] == 100
        assert RATE_LIMITS["enterprise"] == 1000

        # Enterprise should be highest
        assert RATE_LIMITS["enterprise"] > RATE_LIMITS["premium"]
        assert RATE_LIMITS["premium"] > RATE_LIMITS["free"]


class TestSecurityHeaders:
    """Tests for security-related headers."""

    @pytest.mark.asyncio
    async def test_www_authenticate_header_on_missing_key(self):
        """Test WWW-Authenticate header is set on 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key=None)

        assert "WWW-Authenticate" in exc_info.value.headers
        assert exc_info.value.headers["WWW-Authenticate"] == "ApiKey"

    @pytest.mark.asyncio
    async def test_www_authenticate_header_on_invalid_key(self):
        """Test WWW-Authenticate header is set on invalid key."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="invalid_key")

        assert "WWW-Authenticate" in exc_info.value.headers
        assert exc_info.value.headers["WWW-Authenticate"] == "ApiKey"
