"""Unit tests for Redis Semantic Cache and vector similarity thresholding."""

import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache


@pytest.mark.asyncio
async def test_cache_miss_on_empty(mock_semantic_cache: SemanticCache):
    """Verify that an empty cache returns a miss without errors."""
    query_vector = np.random.randn(384).astype(np.float32)
    result = await mock_semantic_cache.get(tenant_id="tenant-1", query_vector=query_vector)

    assert result.hit is False
    assert result.response_text is None
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_cache_hit_exact_match(mock_semantic_cache: SemanticCache):
    """Verify that exact vector produces distance near 0.0 (< 0.08 threshold) and hits."""
    tenant = "tenant-prod-1"
    vec = np.random.randn(384).astype(np.float32)

    await mock_semantic_cache.set(
        tenant_id=tenant,
        query_text="How do I reset my password?",
        query_vector=vec,
        response_text="Click on Account Settings and select Reset Password.",
    )

    result = await mock_semantic_cache.get(tenant_id=tenant, query_vector=vec)

    assert result.hit is True
    assert result.response_text == "Click on Account Settings and select Reset Password."
    assert result.distance is not None
    assert result.distance < 0.08


@pytest.mark.asyncio
async def test_cache_threshold_boundary(mock_semantic_cache: SemanticCache):
    """Verify strict cosine distance threshold boundary (< 0.08 for hit, >= 0.08 for miss)."""
    tenant = "tenant-alpha"
    base_vec = np.zeros(384, dtype=np.float32)
    base_vec[0] = 1.0  # Unit vector along axis 0

    await mock_semantic_cache.set(
        tenant_id=tenant,
        query_text="What is the refund policy?",
        query_vector=base_vec,
        response_text="Refunds are processed within 14 business days.",
    )

    # 1. Perturb slightly: angle theta such that 1 - cos(theta) = 0.04 (< 0.08) -> HIT
    # cos(theta) = 0.96 -> sin(theta) = sqrt(1 - 0.96^2) = 0.28
    near_vec = np.zeros(384, dtype=np.float32)
    near_vec[0] = 0.96
    near_vec[1] = 0.28

    hit_res = await mock_semantic_cache.get(tenant_id=tenant, query_vector=near_vec)
    assert hit_res.hit is True
    assert hit_res.distance is not None
    assert hit_res.distance < 0.08

    # 2. Perturb more: angle theta such that 1 - cos(theta) = 0.15 (>= 0.08) -> MISS
    # cos(theta) = 0.85 -> sin(theta) = sqrt(1 - 0.85^2) = 0.5267
    far_vec = np.zeros(384, dtype=np.float32)
    far_vec[0] = 0.85
    far_vec[1] = 0.5267

    miss_res = await mock_semantic_cache.get(tenant_id=tenant, query_vector=far_vec)
    assert miss_res.hit is False
    assert miss_res.distance is not None
    assert miss_res.distance >= 0.08


@pytest.mark.asyncio
async def test_vector_dimension_mismatch_raises(mock_semantic_cache: SemanticCache):
    """Verify that vector dimensionality mismatches raise ValueError."""
    tenant = "tenant-delta"
    invalid_vec = [0.1] * 128  # Expected 384

    with pytest.raises(ValueError) as exc:
        await mock_semantic_cache.get(tenant_id=tenant, query_vector=invalid_vec)
    assert "Vector shape mismatch" in str(exc.value)


@pytest.mark.asyncio
async def test_tenant_invalidation(mock_semantic_cache: SemanticCache):
    """Verify invalidate_tenant deletes only that tenant's cached entries."""
    vec = np.random.randn(384).astype(np.float32)

    await mock_semantic_cache.set("tenant-purge", "Query 1", vec, "Answer 1")
    await mock_semantic_cache.set("tenant-keep", "Query 1", vec, "Answer 1")

    # Invalidate tenant-purge
    deleted = await mock_semantic_cache.invalidate_tenant("tenant-purge")
    assert deleted >= 1

    # Verify tenant-purge is gone
    res_purged = await mock_semantic_cache.get("tenant-purge", vec)
    assert res_purged.hit is False

    # Verify tenant-keep is intact
    res_keep = await mock_semantic_cache.get("tenant-keep", vec)
    assert res_keep.hit is True
