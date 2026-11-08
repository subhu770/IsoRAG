"""FastAPI dependency injection providers.

Enforces:
- Hard tenant isolation via strict extraction and validation of `X-Tenant-ID`.
- Lifecycle-managed singletons for Redis, Qdrant, embeddings, reranker, pruner, and LLM clients.
- Zero hardcoded secrets, sourcing configuration strictly via Settings.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Header, HTTPException, status
from qdrant_client import AsyncQdrantClient

from src.cache.semantic_cache import SemanticCache
from src.core.config import get_settings

# Strict tenant ID format: alphanumeric with hyphen, underscore, and dot (3-64 chars)
# Prohibits colons, brackets, spaces, quotes, and control chars to avoid RediSearch tag injection.
TENANT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{3,64}$")
TENANT_HEADER_NAME = "X-Tenant-ID"

# Global client cache to prevent connection leak across requests
_REDIS_CLIENT: aioredis.Redis | None = None
_QDRANT_CLIENT: AsyncQdrantClient | None = None
_SEMANTIC_CACHE: SemanticCache | None = None


async def get_tenant_id(
    x_tenant_id: str | None = Header(
        default=None,
        alias=TENANT_HEADER_NAME,
        description="Mandatory multi-tenant isolation identifier",
    ),
) -> str:
    """Extract and strictly validate the tenant ID from request headers.

    Raises:
        HTTPException(400): If header is absent or violates security formatting rules.
    """
    if not x_tenant_id or not x_tenant_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required header: '{TENANT_HEADER_NAME}' is mandatory for tenant isolation.",
        )

    tenant_id = x_tenant_id.strip()
    if not TENANT_ID_REGEX.match(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid '{TENANT_HEADER_NAME}' format: '{tenant_id}'. "
                "Tenant ID must be 3-64 characters long and contain only alphanumeric characters, "
                "underscores, hyphens, and periods."
            ),
        )

    return tenant_id


async def get_redis_client() -> AsyncGenerator[aioredis.Redis, None]:
    """Provide an async Redis client backed by a connection pool."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        settings = get_settings()
        _REDIS_CLIENT = aioredis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            decode_responses=False,
            max_connections=50,
        )
    yield _REDIS_CLIENT


async def get_qdrant_client() -> AsyncGenerator[AsyncQdrantClient, None]:
    """Provide an async Qdrant client connection."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        settings = get_settings()
        if settings.qdrant_url:
            _QDRANT_CLIENT = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )
        else:
            _QDRANT_CLIENT = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                api_key=settings.qdrant_api_key,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )
    yield _QDRANT_CLIENT


async def get_semantic_cache() -> SemanticCache:
    """Provide a singleton instance of SemanticCache."""
    global _SEMANTIC_CACHE, _REDIS_CLIENT
    if _SEMANTIC_CACHE is None:
        settings = get_settings()
        if _REDIS_CLIENT is None:
            _REDIS_CLIENT = aioredis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                password=settings.redis_password,
                ssl=settings.redis_ssl,
                decode_responses=False,
                max_connections=50,
            )
        _SEMANTIC_CACHE = SemanticCache(
            redis_client=_REDIS_CLIENT,
            index_name=settings.redis_semantic_cache_index_name,
            threshold=settings.redis_semantic_cache_threshold,
            vector_dim=settings.redis_semantic_cache_vector_dim,
            default_ttl=settings.redis_semantic_cache_ttl,
        )
    return _SEMANTIC_CACHE


async def close_connections() -> None:
    """Gracefully close all background database connection pools."""
    global _REDIS_CLIENT, _QDRANT_CLIENT
    if _REDIS_CLIENT is not None:
        await _REDIS_CLIENT.aclose()
        _REDIS_CLIENT = None
    if _QDRANT_CLIENT is not None:
        await _QDRANT_CLIENT.close()
        _QDRANT_CLIENT = None
