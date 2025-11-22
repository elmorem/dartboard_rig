"""
Authentication for Dartboard RAG API.

Provides API key validation and user tier management.
"""

import logging
from typing import Optional, Dict
from fastapi import Header, HTTPException
from datetime import datetime

logger = logging.getLogger(__name__)


# API Key Storage (In-memory for now - replace with database in production)
# Format: {api_key: {metadata}}
VALID_API_KEYS: Dict[str, Dict] = {
    "sk_test_123456": {
        "name": "Test User",
        "tier": "free",
        "created_at": "2025-01-01T00:00:00Z",
        "active": True,
    },
    "sk_prod_789012": {
        "name": "Production User",
        "tier": "premium",
        "created_at": "2025-01-01T00:00:00Z",
        "active": True,
    },
}


# Rate limits by tier (requests per minute)
RATE_LIMITS = {
    "free": 10,
    "premium": 100,
    "enterprise": 1000,
}


class APIKeyInfo:
    """API key information."""

    def __init__(self, key: str, metadata: Dict):
        self.key = key
        self.name = metadata.get("name", "Unknown")
        self.tier = metadata.get("tier", "free")
        self.created_at = metadata.get("created_at")
        self.active = metadata.get("active", True)

    @property
    def rate_limit(self) -> int:
        """Get rate limit for this tier."""
        return RATE_LIMITS.get(self.tier, 10)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "tier": self.tier,
            "created_at": self.created_at,
            "active": self.active,
            "rate_limit": self.rate_limit,
        }


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for authentication")
) -> APIKeyInfo:
    """
    Verify API key from request header.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        APIKeyInfo: Key information and metadata

    Raises:
        HTTPException: 401 if key is missing or invalid
    """
    if not x_api_key:
        logger.warning("Request without API key")
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header in your request.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate key
    key_metadata = VALID_API_KEYS.get(x_api_key)
    if not key_metadata:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Check your credentials.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check if key is active
    if not key_metadata.get("active", True):
        logger.warning(f"Inactive API key used: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="API key has been deactivated. Contact support.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_info = APIKeyInfo(x_api_key, key_metadata)
    logger.debug(f"Authenticated: {key_info.name} ({key_info.tier})")

    return key_info


async def verify_api_key_optional(
    x_api_key: Optional[str] = Header(None, description="Optional API key")
) -> Optional[APIKeyInfo]:
    """
    Optional API key verification (for public endpoints).

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        APIKeyInfo if key provided and valid, None otherwise
    """
    if not x_api_key:
        return None

    key_metadata = VALID_API_KEYS.get(x_api_key)
    if not key_metadata or not key_metadata.get("active", True):
        return None

    return APIKeyInfo(x_api_key, key_metadata)


def add_api_key(key: str, name: str, tier: str = "free") -> bool:
    """
    Add a new API key (for management/admin use).

    Args:
        key: API key string
        name: User/organization name
        tier: Tier level (free, premium, enterprise)

    Returns:
        bool: True if added successfully
    """
    if key in VALID_API_KEYS:
        logger.warning(f"Attempted to add duplicate key: {key[:10]}...")
        return False

    VALID_API_KEYS[key] = {
        "name": name,
        "tier": tier,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "active": True,
    }

    logger.info(f"Added new API key for {name} ({tier})")
    return True


def revoke_api_key(key: str) -> bool:
    """
    Revoke (deactivate) an API key.

    Args:
        key: API key to revoke

    Returns:
        bool: True if revoked successfully
    """
    if key not in VALID_API_KEYS:
        logger.warning(f"Attempted to revoke non-existent key: {key[:10]}...")
        return False

    VALID_API_KEYS[key]["active"] = False
    logger.info(f"Revoked API key: {key[:10]}...")
    return True


def get_all_keys() -> Dict[str, Dict]:
    """
    Get all API keys (for admin/debugging).

    Returns:
        Dict mapping keys to metadata
    """
    return {
        key[:10]
        + "...": {
            **metadata,
            "key_prefix": key[:10],
        }
        for key, metadata in VALID_API_KEYS.items()
    }
