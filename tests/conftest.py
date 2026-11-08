"""Pytest configuration, mock fixtures, and test harnesses."""

from typing import Any

import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache
from src.core.config import Settings


class MockRedis:
    """In-memory mock for Redis and RediSearch commands for unit testing."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}
        self.ttls: dict[str, int] = {}

    async def execute_command(self, *args: Any) -> Any:
        cmd = args[0].upper()
        if cmd == "FT.INFO":
            return ["index_name", "mock_idx"]
        elif cmd == "FT.CREATE":
            return b"OK"
        elif cmd == "FT.SEARCH":
            # RediSearch query parser mock
            # Args: FT.SEARCH index_name query PARAMS 4 t_id tenant query_vec bytes ...
            _index_name = args[1]
            _query_str = args[2]

            # Find params
            params_idx = -1
            for i, a in enumerate(args):
                if str(a).upper() == "PARAMS":
                    params_idx = i
                    break

            target_tenant = None
            query_vec_bytes = None
            if params_idx != -1:
                # args: PARAMS, num, key1, val1, key2, val2
                p_dict = {}
                p_count = int(args[params_idx + 1])
                for j in range(0, p_count * 2, 2):
                    pk = args[params_idx + 2 + j]
                    pv = args[params_idx + 3 + j]
                    p_dict[pk] = pv
                target_tenant = p_dict.get("t_id")
                query_vec_bytes = p_dict.get("query_vec")

            # Search in self.store
            hits = []
            for k, val in self.store.items():
                if val.get("tenant_id") == target_tenant:
                    stored_bytes = val.get("vector")
                    # compute cosine distance
                    if stored_bytes and query_vec_bytes:
                        v1 = np.frombuffer(query_vec_bytes, dtype=np.float32)
                        v2 = np.frombuffer(stored_bytes, dtype=np.float32)
                        dot = float(np.dot(v1, v2))
                        n1 = float(np.linalg.norm(v1))
                        n2 = float(np.linalg.norm(v2))
                        cos_sim = dot / (n1 * n2) if (n1 * n2) > 0 else 0.0
                        dist = max(0.0, 1.0 - cos_sim)
                    else:
                        dist = 0.0
                    hits.append((dist, k, val))

            if not hits:
                return [0]

            # Sort by distance
            hits.sort(key=lambda x: x[0])
            best_dist, best_key, best_val = hits[0]
            fields = [
                b"query_text",
                best_val.get("query_text", "").encode("utf-8"),
                b"response_text",
                best_val.get("response_text", "").encode("utf-8"),
                b"metadata",
                best_val.get("metadata", "{}").encode("utf-8"),
                b"created_at",
                str(best_val.get("created_at", 0)).encode("utf-8"),
                b"vector_distance",
                str(round(best_dist, 6)).encode("utf-8"),
            ]
            return [1, best_key.encode("utf-8"), fields]
        return b"OK"

    def pipeline(self, transaction: bool = True) -> "MockPipeline":
        return MockPipeline(self)

    async def scan(
        self, cursor: int = 0, match: str = "*", count: int = 10
    ) -> tuple[int, list[str]]:
        import fnmatch

        matched = [k for k in self.store.keys() if fnmatch.fnmatch(k, match)]
        return 0, matched

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                count += 1
        return count

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


class MockPipeline:
    def __init__(self, mock_redis: MockRedis) -> None:
        self.mock_redis = mock_redis
        self.ops = []

    def hset(self, key: str, mapping: dict[str, Any]) -> "MockPipeline":
        self.ops.append(("hset", key, mapping))
        return self

    def expire(self, key: str, seconds: int) -> "MockPipeline":
        self.ops.append(("expire", key, seconds))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for op in self.ops:
            if op[0] == "hset":
                _, key, mapping = op
                self.mock_redis.store[key] = mapping
                results.append(len(mapping))
            elif op[0] == "expire":
                _, key, seconds = op
                self.mock_redis.ttls[key] = seconds
                results.append(True)
        return results


@pytest.fixture
def mock_redis() -> MockRedis:
    return MockRedis()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app_name="IsoRAG-Test",
        redis_semantic_cache_threshold=0.08,
        redis_semantic_cache_vector_dim=384,
        qdrant_collection_name="test_isorag_collection",
        openai_api_key="mock-test-key",
    )


@pytest.fixture
def mock_semantic_cache(mock_redis: MockRedis, test_settings: Settings) -> SemanticCache:
    return SemanticCache(
        redis_client=mock_redis,  # type: ignore
        threshold=0.08,
        vector_dim=384,
        default_ttl=3600,
    )
