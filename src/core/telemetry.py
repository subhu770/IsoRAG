"""Latency telemetry and pipeline stage performance logging."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger("isorag.telemetry")


class PipelineTelemetry(BaseModel):
    """Telemetry report capturing per-stage execution latencies and metadata."""

    trace_id: str = Field(default_factory=lambda: f"trace-{uuid4().hex[:12]}")
    tenant_id: str
    cache_hit: bool = False
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    tokens_compressed: int | None = None
    compression_ratio: float | None = None
    error: str | None = None

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Record stage execution duration in milliseconds."""
        self.stage_latencies_ms[stage_name] = round(duration_ms, 2)

    def finish(self, start_time: float) -> None:
        """Finalize total request latency and emit structured log."""
        self.total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "pipeline_telemetry trace_id=%s tenant_id=%s cache_hit=%s total_ms=%.2f stages=%s",
            self.trace_id,
            self.tenant_id,
            self.cache_hit,
            self.total_latency_ms,
            self.stage_latencies_ms,
        )


class StageTimer:
    """Async context manager to benchmark individual RAG pipeline stages."""

    def __init__(self, stage_name: str, telemetry: PipelineTelemetry):
        self.stage_name = stage_name
        self.telemetry = telemetry
        self.start_time: float = 0.0

    async def __aenter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.telemetry.record_stage(self.stage_name, duration_ms)


@asynccontextmanager
async def trace_stage(stage_name: str, telemetry: PipelineTelemetry) -> AsyncGenerator[None, None]:
    """Helper generator for tracing a pipeline stage cleanly."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        telemetry.record_stage(stage_name, duration_ms)
