"""Tenant administration endpoints: cache eviction and vector data purging."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.api.dependencies import get_qdrant_client, get_semantic_cache, get_tenant_id
from src.cache.semantic_cache import SemanticCache
from src.core.config import Settings, get_settings

logger = logging.getLogger("isorag.api.admin")

router = APIRouter(prefix="/tenant", tags=["Tenant Administration"])


class CacheInvalidateResponse(BaseModel):
    """Response returned upon tenant cache eviction."""

    status: str = "success"
    tenant_id: str
    evicted_keys_count: int


class TenantDataPurgeResponse(BaseModel):
    """Response returned upon purging tenant vector database records."""

    status: str = "success"
    tenant_id: str
    message: str


class TenantInfoResponse(BaseModel):
    """Diagnostic info for the calling tenant."""

    tenant_id: str
    isolation_mode: str = "hard_payload_indexing"
    collection_name: str
    semantic_cache_threshold: float
    vector_dimension: int


@router.post(
    "/cache/invalidate",
    response_model=CacheInvalidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Purge all semantic cache entries for tenant",
)
async def invalidate_tenant_cache(
    tenant_id: str = Depends(get_tenant_id),
    semantic_cache: SemanticCache = Depends(get_semantic_cache),
) -> CacheInvalidateResponse:
    """Evict all cached question-answer entries belonging strictly to the calling tenant."""
    deleted_count = await semantic_cache.invalidate_tenant(tenant_id)
    return CacheInvalidateResponse(
        status="success",
        tenant_id=tenant_id,
        evicted_keys_count=deleted_count,
    )


@router.delete(
    "/data",
    response_model=TenantDataPurgeResponse,
    status_code=status.HTTP_200_OK,
    summary="GDPR Purge: permanently delete all tenant vector store documents",
)
async def purge_tenant_data(
    tenant_id: str = Depends(get_tenant_id),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
    settings: Settings = Depends(get_settings),
) -> TenantDataPurgeResponse:
    """Permanently delete all documents, embeddings, and payloads for the calling tenant in Qdrant."""
    tenant_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="tenant_id",
                match=qmodels.MatchValue(value=tenant_id),
            )
        ]
    )

    try:
        await qdrant.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=tenant_filter,
            wait=True,
        )
        logger.info("Successfully purged all Qdrant vectors for tenant: %s", tenant_id)
        return TenantDataPurgeResponse(
            status="success",
            tenant_id=tenant_id,
            message=f"All vector records and payloads for tenant '{tenant_id}' were permanently deleted.",
        )
    except Exception as e:
        logger.error("Failed to purge Qdrant data for tenant '%s': %s", tenant_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to purge data for tenant '{tenant_id}': {str(e)}",
        ) from e


@router.get(
    "/info",
    response_model=TenantInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch tenant metadata and isolation settings",
)
async def get_tenant_info(
    tenant_id: str = Depends(get_tenant_id),
    settings: Settings = Depends(get_settings),
) -> TenantInfoResponse:
    """Return runtime isolation and configuration parameters for tenant."""
    return TenantInfoResponse(
        tenant_id=tenant_id,
        collection_name=settings.qdrant_collection_name,
        semantic_cache_threshold=settings.redis_semantic_cache_threshold,
        vector_dimension=settings.dense_vector_dim,
    )
