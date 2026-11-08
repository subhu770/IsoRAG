"""Tests for Mock LLM Mode in src/generator/llm_client.py."""

import time

import pytest

from src.core.config import Settings
from src.generator.llm_client import LLMClient
from src.retrieval.hybrid_retriever import DocumentChunk, ScoredChunk


@pytest.fixture
def sample_chunks() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=DocumentChunk(
                doc_id="sec-chunk-101",
                tenant_id="tenant-alpha",
                content="Tenants authenticate using API keys or JWT tokens signed by an internal KMS.",
                metadata={"classification": "confidential"},
            ),
            score=0.96,
            retrieval_method="cross_encoder_rerank",
        ),
        ScoredChunk(
            chunk=DocumentChunk(
                doc_id="sec-chunk-102",
                tenant_id="tenant-alpha",
                content="Data at rest is encrypted with AES-256 GCM across all storage tiers.",
                metadata={"classification": "internal"},
            ),
            score=0.89,
            retrieval_method="cross_encoder_rerank",
        ),
    ]


@pytest.mark.asyncio
async def test_mock_llm_mode_with_blank_api_key(sample_chunks: list[ScoredChunk], monkeypatch):
    """Verify Mock LLM Mode activates when SIMULATION_MODE=true without an API key."""
    monkeypatch.setenv("SIMULATION_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = Settings(openai_api_key=None, simulation_mode=True)
    client = LLMClient(settings=settings)

    assert client.is_simulation_mode is True

    start_time = time.perf_counter()
    response = await client.generate(
        query="How is data encrypted at rest?",
        context="Tenants authenticate using API keys. Data at rest is encrypted with AES-256 GCM.",
        scored_chunks=sample_chunks,
    )
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Verify artificial sleep (between 50ms and 140ms)
    assert duration_ms >= 45.0, f"Expected simulated sleep >= 45ms, got {duration_ms:.2f}ms"

    # Verify strict schema attributes
    assert response.answer is not None
    assert len(response.citations) == 2
    assert "sec-chunk-101" in response.citations
    assert "sec-chunk-102" in response.citations
    assert "simulated" in response.model
    assert response.raw_prompt_tokens is not None and response.raw_prompt_tokens > 0
    assert response.completion_tokens is not None and response.completion_tokens > 0

    # Verify deterministic content constructed from context
    assert "[sec-chunk-101]" in response.answer
    assert "[sec-chunk-102]" in response.answer
    assert "AES-256 GCM" in response.answer


@pytest.mark.asyncio
async def test_mock_llm_mode_empty_context():
    """Verify Mock LLM Mode handles empty context gracefully."""
    settings = Settings(openai_api_key=None, simulation_mode=True)
    client = LLMClient(settings=settings)

    response = await client.generate(
        query="What is the meaning of life?",
        context="",
        scored_chunks=[],
    )

    assert "do not have sufficient information" in response.answer
    assert len(response.citations) == 0


@pytest.mark.asyncio
async def test_mock_llm_streaming(sample_chunks: list[ScoredChunk], monkeypatch):
    """Verify Mock LLM streaming yields token chunks with simulated latency."""
    monkeypatch.setenv("SIMULATION_MODE", "true")
    settings = Settings(openai_api_key=None, simulation_mode=True)
    client = LLMClient(settings=settings)

    streamed_tokens: list[str] = []
    start_time = time.perf_counter()

    async for token in client.stream_generate(
        query="How is authentication handled?",
        context="Tenants authenticate using API keys.",
        scored_chunks=sample_chunks,
    ):
        streamed_tokens.append(token)

    total_time_ms = (time.perf_counter() - start_time) * 1000

    assert len(streamed_tokens) > 5
    assert total_time_ms > 20.0
    combined_text = "".join(streamed_tokens)
    assert "[sec-chunk-101]" in combined_text
