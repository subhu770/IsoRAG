"""End-to-end integration tests for IsoRAG API endpoints.

Verifies:
- Document batch ingestion under tenant isolation.
- Full query pipeline execution (Hybrid retrieval -> Rerank -> Prune -> Response).
- Semantic cache hit on repeated query.
- Strict cross-tenant isolation (Tenant B cannot query Tenant A's documents or cache).
- Health and diagnostic probes.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_qdrant_client, get_redis_client, get_semantic_cache
from src.main import app


@pytest.fixture
def mock_qdrant_e2e():
    """Mock Qdrant client supporting simulated multi-tenant storage."""
    client = AsyncMock()
    # In-memory document storage: key is tenant_id, value is list of points
    storage: dict[str, list[dict[str, Any]]] = {}

    async def mock_get_collections():
        m = MagicMock()
        col = MagicMock()
        col.name = "isorag_documents"
        m.collections = [col]
        return m

    async def mock_upsert(collection_name: str, points: list[Any], wait: bool = True):
        for pt in points:
            tid = pt.payload["tenant_id"]
            if tid not in storage:
                storage[tid] = []
            storage[tid].append(
                {
                    "id": pt.id,
                    "payload": pt.payload,
                    "vector": pt.vector,
                }
            )
        return MagicMock()

    async def mock_search(
        collection_name: str,
        query_vector: Any,
        query_filter: Any,
        limit: int = 10,
        with_payload: bool = True,
    ):
        # Extract target tenant from filter
        target_tenant = None
        if query_filter and query_filter.must:
            for cond in query_filter.must:
                if cond.key == "tenant_id":
                    target_tenant = cond.match.value

        results = []
        tenant_points = storage.get(target_tenant, [])
        for pt in tenant_points[:limit]:
            hit = MagicMock()
            hit.id = pt["id"]
            hit.score = 0.92
            hit.payload = pt["payload"]
            results.append(hit)
        return results

    async def mock_query_points(
        collection_name: str,
        query: Any = None,
        query_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
        **kwargs: Any,
    ):
        points = await mock_search(collection_name, query, query_filter, limit, with_payload)
        resp = MagicMock()
        resp.points = points
        return resp

    client.get_collections = mock_get_collections
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.upsert = mock_upsert
    client.search = mock_search
    client.query_points = mock_query_points
    return client


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Verify /health endpoint returns HTTP 200 and connectivity status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["app"] == "IsoRAG"


@pytest.mark.asyncio
async def test_e2e_ingest_and_isolated_query(
    mock_redis,
    mock_semantic_cache,
    mock_qdrant_e2e,
):
    """Verify complete end-to-end ingestion, query, caching, and tenant isolation."""
    # Override FastAPI dependencies with test mocks
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[get_semantic_cache] = lambda: mock_semantic_cache
    app.dependency_overrides[get_qdrant_client] = lambda: mock_qdrant_e2e

    tenant_a = "tenant-alpha-001"
    tenant_b = "tenant-beta-002"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Ingest document for Tenant A
        ingest_payload = {
            "documents": [
                {
                    "doc_id": "alpha-doc-1",
                    "content": "Tenant Alpha proprietary algorithm uses custom gradient masking.",
                    "metadata": {"dept": "core-ai"},
                }
            ]
        }
        res_ingest = await client.post(
            "/api/v1/ingest",
            headers={"X-Tenant-ID": tenant_a},
            json=ingest_payload,
        )
        assert res_ingest.status_code == 201
        data_ingest = res_ingest.json()
        assert data_ingest["status"] == "success"
        assert data_ingest["tenant_id"] == tenant_a
        assert data_ingest["ingested_count"] == 1

        # 2. Query as Tenant A (Cold query -> Cache Miss -> Hybrid Search -> Synthesize)
        query_payload = {
            "query": "What proprietary algorithm does our team use?",
            "top_k": 3,
        }
        res_query_a1 = await client.post(
            "/api/v1/query",
            headers={"X-Tenant-ID": tenant_a},
            json=query_payload,
        )
        assert res_query_a1.status_code == 200, f"Query failed: {res_query_a1.text}"
        data_query_a1 = res_query_a1.json()
        assert data_query_a1["tenant_id"] == tenant_a
        assert data_query_a1["cache_hit"] is False
        assert len(data_query_a1["sources"]) == 1
        assert data_query_a1["sources"][0]["doc_id"] == "alpha-doc-1"
        assert "alpha-doc-1" in data_query_a1["citations"]

        # 3. Query as Tenant A second time (Warm query -> Cache Hit!)
        res_query_a2 = await client.post(
            "/api/v1/query",
            headers={"X-Tenant-ID": tenant_a},
            json=query_payload,
        )
        assert res_query_a2.status_code == 200
        data_query_a2 = res_query_a2.json()
        assert data_query_a2["cache_hit"] is True
        assert data_query_a2["answer"] == data_query_a1["answer"]
        assert data_query_a2["telemetry"]["cache_hit"] is True

        # 4. Query as Tenant B with identical question -> MUST NOT return Tenant A's data or cache!
        res_query_b = await client.post(
            "/api/v1/query",
            headers={"X-Tenant-ID": tenant_b},
            json=query_payload,
        )
        assert res_query_b.status_code == 200
        data_query_b = res_query_b.json()
        assert data_query_b["tenant_id"] == tenant_b
        assert data_query_b["cache_hit"] is False
        # Tenant B has 0 documents ingested, so sources must be empty!
        assert len(data_query_b["sources"]) == 0
        assert "alpha-doc-1" not in data_query_b["citations"]

    # Clean up overrides
    app.dependency_overrides.clear()
