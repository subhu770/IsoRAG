"""Production-grade Semantic Cache backed by Redis Stack (RediSearch).

Features:
- Sub-millisecond vector similarity search using RediSearch FLAT/HNSW index.
- Strict multi-tenant isolation via RediSearch TAG filtering and keyspace prefixing.
- Cosine distance thresholding (< 0.08 for cache hit).
- Configurable per-tenant TTL and bulk tenant invalidation.
- Graceful degradation: falls back to cache miss on connection/query failures without blocking retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import numpy as np
import redis.asyncio as aioredis
from pydantic import BaseModel, Field
from redis.exceptions import ConnectionError, RedisError, ResponseError

logger = logging.getLogger("isorag.cache.semantic")


class CacheHitResult(BaseModel):
    """Result returned by a semantic cache lookup."""

    hit: bool = Field(
        description="True if a semantically equivalent query was found within threshold"
    )
    query_text: str | None = Field(default=None, description="Original cached query string")
    response_text: str | None = Field(default=None, description="Cached synthesized response")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with cache entry"
    )
    distance: float | None = Field(
        default=None, description="Measured cosine distance to the query vector"
    )
    key: str | None = Field(default=None, description="Redis key of the matched entry")
    latency_ms: float = Field(default=0.0, description="Lookup execution time in ms")


class SemanticCache:
    """Manages multi-tenant semantic caching in Redis with RediSearch vector indexes."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        index_name: str = "isorag:semantic_cache_idx",
        threshold: float = 0.08,
        vector_dim: int = 384,
        default_ttl: int = 86400,
    ) -> None:
        """Initialize SemanticCache instance.

        Args:
            redis_client: Async Redis connection or connection pool.
            index_name: Identifier for the RediSearch index.
            threshold: Maximum cosine distance to classify as hit (default < 0.08).
            vector_dim: Dimensionality of query vector embeddings.
            default_ttl: TTL for cache keys in seconds.
        """
        self.redis = redis_client
        self.index_name = index_name
        self.threshold = threshold
        self.vector_dim = vector_dim
        self.default_ttl = default_ttl
        self._index_initialized = False

    async def ensure_index(self) -> None:
        """Ensure the RediSearch schema with tenant tag and vector field exists.

        Executes idempotent FT.CREATE if index is absent.
        """
        if self._index_initialized:
            return

        try:
            # Check if index already exists
            await self.redis.execute_command("FT.INFO", self.index_name)
            self._index_initialized = True
            logger.debug("RediSearch semantic cache index '%s' is ready.", self.index_name)
            return
        except ResponseError as e:
            # If error is not 'Unknown index name', re-raise
            if "unknown index name" not in str(e).lower() and "no such index" not in str(e).lower():
                logger.error("Unexpected error querying FT.INFO for '%s': %s", self.index_name, e)
                raise

        # Create index schema with Tenant TAG and Vector FLOAT32 COSINE
        try:
            cmd = [
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                "cache:",
                "SCHEMA",
                "tenant_id",
                "TAG",
                "query_text",
                "TEXT",
                "response_text",
                "TEXT",
                "metadata",
                "TEXT",
                "created_at",
                "NUMERIC",
                "vector",
                "VECTOR",
                "FLAT",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.vector_dim),
                "DISTANCE_METRIC",
                "COSINE",
            ]
            await self.redis.execute_command(*cmd)
            self._index_initialized = True
            logger.info(
                "Created RediSearch semantic cache index '%s' (dim=%d, metric=COSINE).",
                self.index_name,
                self.vector_dim,
            )
        except ResponseError as create_err:
            if "index already exists" in str(create_err).lower():
                self._index_initialized = True
            else:
                logger.error(
                    "Failed to create RediSearch index '%s': %s", self.index_name, create_err
                )
                raise

    def _normalize_vector(self, vector: list[float] | np.ndarray) -> bytes:
        """Convert input vector to normalized float32 byte array.

        Args:
            vector: Float array or list representing query embedding.

        Returns:
            Byte string compatible with RediSearch vector format.
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.vector_dim:
            raise ValueError(
                f"Vector shape mismatch: expected 1D array of dim {self.vector_dim}, got {arr.shape}"
            )
        # RediSearch COSINE distance assumes unit vectors or computes cosine similarity.
        # Normalizing improves numerical stability and consistency.
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32).tobytes()

    @staticmethod
    def _generate_cache_key(tenant_id: str, query_text: str) -> str:
        """Construct tenant-isolated Redis key using SHA-256 hash."""
        digest = hashlib.sha256(query_text.strip().lower().encode("utf-8")).hexdigest()[:24]
        return f"cache:{tenant_id}:{digest}"

    async def get(
        self,
        tenant_id: str,
        query_vector: list[float] | np.ndarray,
        custom_threshold: float | None = None,
    ) -> CacheHitResult:
        """Perform KNN vector search within tenant partition to locate cached answers.

        Args:
            tenant_id: Mandatory isolated tenant identifier.
            query_vector: Query dense embedding vector.
            custom_threshold: Optional override for cosine distance cutoff.

        Returns:
            CacheHitResult detailing hit status, response, and measured distance.
        """
        start_time = time.perf_counter()
        active_threshold = custom_threshold if custom_threshold is not None else self.threshold

        # Validate vector dimensionality before entering Redis try-block
        vector_bytes = self._normalize_vector(query_vector)

        try:
            await self.ensure_index()

            # RediSearch query: Filter strictly by tenant tag and fetch 1 nearest neighbor
            # Tag syntax requires escaping special characters or binding parameters.
            # Here we bind tenant_id to prevent injection vulnerabilities.
            query_str = "(@tenant_id:{$t_id})=>[KNN 1 @vector $query_vec AS vector_distance]"
            search_cmd = [
                "FT.SEARCH",
                self.index_name,
                query_str,
                "PARAMS",
                "4",
                "t_id",
                tenant_id,
                "query_vec",
                vector_bytes,
                "SORTBY",
                "vector_distance",
                "ASC",
                "RETURN",
                "5",
                "query_text",
                "response_text",
                "metadata",
                "created_at",
                "vector_distance",
                "DIALECT",
                "2",
                "LIMIT",
                "0",
                "1",
            ]

            raw_result = await self.redis.execute_command(*search_cmd)
            lookup_ms = (time.perf_counter() - start_time) * 1000

            if not raw_result or raw_result[0] == 0:
                return CacheHitResult(hit=False, latency_ms=lookup_ms)

            # Parse RediSearch dialect 2 return array: [total, doc_key, [field, val, ...]]
            doc_key = raw_result[1]
            if isinstance(doc_key, bytes):
                doc_key = doc_key.decode("utf-8")

            fields_list = raw_result[2]
            parsed_fields: dict[str, Any] = {}
            for i in range(0, len(fields_list), 2):
                k = (
                    fields_list[i].decode("utf-8")
                    if isinstance(fields_list[i], bytes)
                    else str(fields_list[i])
                )
                v = (
                    fields_list[i + 1].decode("utf-8")
                    if isinstance(fields_list[i + 1], bytes)
                    else fields_list[i + 1]
                )
                parsed_fields[k] = v

            distance = float(parsed_fields.get("vector_distance", 1.0))

            # Cosine distance threshold validation: distance < threshold constitutes a HIT
            if distance < active_threshold:
                metadata: dict[str, Any] = {}
                meta_raw = parsed_fields.get("metadata")
                if meta_raw:
                    try:
                        metadata = json.loads(meta_raw)
                    except Exception:
                        metadata = {"raw": meta_raw}

                logger.debug(
                    "Semantic cache HIT for tenant '%s' (dist=%.4f < threshold=%.4f)",
                    tenant_id,
                    distance,
                    active_threshold,
                )
                return CacheHitResult(
                    hit=True,
                    query_text=parsed_fields.get("query_text"),
                    response_text=parsed_fields.get("response_text"),
                    metadata=metadata,
                    distance=distance,
                    key=doc_key,
                    latency_ms=lookup_ms,
                )

            logger.debug(
                "Semantic cache MISS for tenant '%s' (dist=%.4f >= threshold=%.4f)",
                tenant_id,
                distance,
                active_threshold,
            )
            return CacheHitResult(
                hit=False,
                distance=distance,
                latency_ms=lookup_ms,
            )

        except (ConnectionError, RedisError) as e:
            lookup_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Redis error during semantic cache lookup for tenant '%s': %s. Degraded to cache miss.",
                tenant_id,
                str(e),
            )
            return CacheHitResult(hit=False, latency_ms=lookup_ms)
        except Exception as ex:
            lookup_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Unexpected failure in semantic cache lookup for tenant '%s': %s",
                tenant_id,
                str(ex),
                exc_info=True,
            )
            return CacheHitResult(hit=False, latency_ms=lookup_ms)

    async def set(
        self,
        tenant_id: str,
        query_text: str,
        query_vector: list[float] | np.ndarray,
        response_text: str,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Store synthesized response and embedding into tenant-partitioned cache.

        Args:
            tenant_id: Isolated tenant ID.
            query_text: Natural language user query.
            query_vector: Dense embedding vector for semantic matching.
            response_text: Synthesized answer string.
            metadata: Associated retrieval provenance or context metadata.
            ttl: TTL in seconds (falls back to default_ttl).

        Returns:
            True if cached successfully, False otherwise.
        """
        # Validate vector dimensionality before entering Redis try-block
        vector_bytes = self._normalize_vector(query_vector)
        key = self._generate_cache_key(tenant_id, query_text)
        effective_ttl = ttl if ttl is not None else self.default_ttl

        try:
            await self.ensure_index()

            payload = {
                "tenant_id": tenant_id,
                "query_text": query_text,
                "response_text": response_text,
                "metadata": json.dumps(metadata or {}),
                "created_at": time.time(),
                "vector": vector_bytes,
            }

            pipe = self.redis.pipeline(transaction=True)
            pipe.hset(key, mapping=payload)
            if effective_ttl > 0:
                pipe.expire(key, effective_ttl)
            await pipe.execute()

            logger.debug(
                "Cached response for tenant '%s' at key '%s' (TTL=%ds)",
                tenant_id,
                key,
                effective_ttl,
            )
            return True
        except (ConnectionError, RedisError) as re:
            logger.warning("Failed to store semantic cache entry in Redis: %s", str(re))
            return False
        except Exception as e:
            logger.error("Unexpected error storing semantic cache entry: %s", str(e), exc_info=True)
            return False

    async def invalidate_tenant(self, tenant_id: str) -> int:
        """Evict all cached entries belonging strictly to the specified tenant.

        Args:
            tenant_id: Tenant to purge from cache.

        Returns:
            Number of evicted cache entries.
        """
        pattern = f"cache:{tenant_id}:*"
        deleted_count = 0
        cursor = 0
        try:
            while True:
                cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    deleted = await self.redis.delete(*keys)
                    deleted_count += deleted
                if cursor == 0:
                    break
            logger.info("Evicted %d cache entries for tenant '%s'", deleted_count, tenant_id)
            return deleted_count
        except RedisError as e:
            logger.error("Failed to invalidate cache for tenant '%s': %s", tenant_id, e)
            return deleted_count
