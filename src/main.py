"""IsoRAG FastAPI Application Entrypoint."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import close_connections, get_redis_client, get_semantic_cache
from src.api.v1 import api_v1_router
from src.core.config import get_settings

# Configure structured logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("isorag.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager for connection setup and graceful teardown."""
    logger.info(
        "Initializing IsoRAG engine services (App: %s, Env: %s)...",
        settings.app_name,
        settings.app_env,
    )
    try:
        # Pre-initialize Redis semantic cache index
        cache = await get_semantic_cache()
        await cache.ensure_index()
    except Exception as e:
        logger.warning("Startup index warm-up warning (will retry on first request): %s", e)

    yield

    logger.info("Shutting down IsoRAG engine: closing active connection pools...")
    await close_connections()
    logger.info("IsoRAG shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-grade, multi-tenant RAG engine with hard isolation and semantic caching.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(api_v1_router, prefix="/api")


@app.get("/health", tags=["Diagnostics"], status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Readiness and health check probing Redis and Qdrant status."""
    health_status: dict[str, str] = {
        "status": "healthy",
        "app": settings.app_name,
        "env": settings.app_env,
    }
    # Check Redis
    try:
        async for redis in get_redis_client():
            await redis.ping()
            health_status["redis"] = "connected"
            break
    except Exception as re:
        health_status["redis"] = f"unhealthy: {str(re)}"
        health_status["status"] = "degraded"

    return health_status


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "engine": "IsoRAG",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


def run_server() -> None:
    """CLI entrypoint to start the application with Uvicorn."""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
