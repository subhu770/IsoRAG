"""Pydantic v2 schemas for IsoRAG API endpoints."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.core.telemetry import PipelineTelemetry


class IngestDocument(BaseModel):
    """Single document chunk to be ingested."""

    doc_id: str = Field(default_factory=lambda: f"doc-{uuid4().hex[:8]}")
    content: str = Field(..., min_length=1, description="Raw text content of the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom payload metadata")


class IngestRequest(BaseModel):
    """Batch document ingestion request."""

    documents: list[IngestDocument] = Field(..., min_length=1, max_length=500)


class IngestResponse(BaseModel):
    """Batch document ingestion response."""

    status: str = "success"
    tenant_id: str
    ingested_count: int
    duration_ms: float


class QueryRequest(BaseModel):
    """User query request payload."""

    query: str = Field(..., min_length=1, max_length=2048, description="Natural language question")
    top_k: int | None = Field(
        default=5, ge=1, le=50, description="Number of source documents to retrieve"
    )
    bypass_cache: bool = Field(
        default=False, description="Force retrieval, bypassing semantic cache"
    )


class SourceDocument(BaseModel):
    """Retrieved source citation item."""

    doc_id: str
    content: str
    score: float
    retrieval_method: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """End-to-end query response with synthesis and provenance."""

    tenant_id: str
    query: str
    answer: str
    cache_hit: bool
    citations: list[str] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)
    telemetry: PipelineTelemetry
