"""Tenant-isolated sliding-window rate limiter backed by Redis.

Guarantees:
- Fair resource allocation across tenants, preventing "noisy neighbor" starvation.
- Strict namespace isolation per tenant: keys formatted as `ratelimit:{tenant_id}`.
- Graceful degradation: permits traffic if Redis connectivity is degraded.
"""

from __future__ import annotations

import logging
import time

import redis.asyncio as aioredis
from fastapi import HTTPException, status

from src.core.config import Settings, get_settings

logger = logging.getLogger("isorag.core.rate_limiter")


class TenantRateLimiter:
    """Sliding-window rate limiter per tenant using Redis sorted sets (ZSET)."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        requests_per_minute: int = 120,
        settings: Settings | None = None,
    ) -> None:
        self.redis = redis_client
        self.rpm = requests_per_minute
        self.window_seconds = 60
        self.settings = settings or get_settings()

    async def check_rate_limit(self, tenant_id: str) -> None:
        """Enforce rate limits per tenant.

        Raises:
            HTTPException(429): When request rate exceeds configured RPM.
        """
        key = f"ratelimit:{tenant_id}"
        now = time.time()
        window_start = now - self.window_seconds

        try:
            pipe = self.redis.pipeline(transaction=True)
            # Remove timestamps outside current sliding window
            pipe.zremrangebyscore(key, 0, window_start)
            # Count remaining timestamps in window
            pipe.zcard(key)
            # Add current request timestamp
            pipe.zadd(key, {str(now): now})
            # Set TTL to auto-expire idle keys
            pipe.expire(key, self.window_seconds + 5)

            results = await pipe.execute()
            current_requests = results[1]

            if current_requests >= self.rpm:
                logger.warning(
                    "Rate limit exceeded for tenant '%s' (%d/%d rpm)",
                    tenant_id,
                    current_requests,
                    self.rpm,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Tenant '{tenant_id}' exceeded rate limit of {self.rpm} req/min. "
                        "Please retry after window reset."
                    ),
                    headers={"Retry-After": "10"},
                )
        except (aioredis.ConnectionError, aioredis.RedisError) as e:
            logger.warning("Redis unavailable for rate limiting: %s. Permitting request.", e)
            return
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Unexpected error in tenant rate limiter: %s", e)
            return
