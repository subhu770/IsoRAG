"""V1 API Router combining ingestion and query endpoints."""

from fastapi import APIRouter

from src.api.v1.ingest import router as ingest_router
from src.api.v1.query import router as query_router
from src.api.v1.tenant_admin import router as tenant_admin_router

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(ingest_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(tenant_admin_router)

__all__ = ["api_v1_router"]
