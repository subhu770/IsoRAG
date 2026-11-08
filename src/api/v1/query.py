"""Query execution pipeline: Semantic Cache -> Hybrid Search -> Rerank -> Prune -> Response."""

import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from qdrant_client import AsyncQdrantClient

from src.api.dependencies import (
    get_qdrant_client,
    get_semantic_cache,
    get_tenant_id,
)
from src.api.v1.schemas import QueryRequest, QueryResponse, SourceDocument
from src.cache.semantic_cache import SemanticCache
from src.compression.pruner import ContextPruner
from src.core.config import Settings, get_settings
from src.core.telemetry import PipelineTelemetry, trace_stage
from src.generator.llm_client import LLMClient
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger("isorag.api.query")
router = APIRouter(prefix="/query", tags=["Query Pipeline"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-tenant semantic RAG query execution",
)
async def query_pipeline(
    request: QueryRequest,
    tenant_id: str = Depends(get_tenant_id),
    semantic_cache: SemanticCache = Depends(get_semantic_cache),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Execute end-to-end RAG query pipeline.

    Flow:
    1. Embed query vector.
    2. Check Redis RediSearch semantic cache (cosine distance < 0.08 = instant hit).
    3. On cache miss: execute hybrid retrieval (Dense Qdrant + Sparse BM25 + RRF).
    4. Cross-encoder rerank candidates using bge-reranker-base.
    5. Prune and compress context using LLMLingua-2.
    6. Synthesize response via LLM client.
    7. Populate semantic cache for tenant.
    """
    total_start_time = time.perf_counter()
    telemetry = PipelineTelemetry(tenant_id=tenant_id)

    retriever = HybridRetriever(qdrant_client=qdrant, settings=settings)
    reranker = Reranker(settings=settings)
    pruner = ContextPruner(settings=settings)
    llm = LLMClient(settings=settings)

    try:
        # Step 1: Compute dense embedding vector
        async with trace_stage("dense_embedding", telemetry):
            query_vector = await retriever.generate_dense_embedding(request.query)

        # Step 2: Semantic Cache Lookup
        if not request.bypass_cache:
            async with trace_stage("semantic_cache_lookup", telemetry):
                cache_result = await semantic_cache.get(
                    tenant_id=tenant_id,
                    query_vector=query_vector,
                )

            if cache_result.hit and cache_result.response_text:
                telemetry.cache_hit = True
                telemetry.finish(total_start_time)
                return QueryResponse(
                    tenant_id=tenant_id,
                    query=request.query,
                    answer=cache_result.response_text,
                    cache_hit=True,
                    citations=cache_result.metadata.get("citations", []),
                    sources=[],
                    telemetry=telemetry,
                )

        # Step 3: Hybrid Retrieval (Qdrant Dense + Sparse BM25 + RRF)
        async with trace_stage("hybrid_retrieval", telemetry):
            fused_candidates = await retriever.hybrid_search(
                tenant_id=tenant_id,
                query=request.query,
                query_dense=query_vector,
                top_k=settings.hybrid_top_k,
            )

        # Step 4: Cross-Encoder Reranking
        async with trace_stage("cross_encoder_rerank", telemetry):
            reranked_chunks = await reranker.rerank(
                query=request.query,
                candidates=fused_candidates,
                top_k=request.top_k,
            )

        # Step 5: Context Compression / Pruning
        async with trace_stage("context_pruning", telemetry):
            compressed_context = await pruner.prune(
                query=request.query,
                chunks=reranked_chunks,
            )
            telemetry.tokens_compressed = compressed_context.compressed_tokens
            telemetry.compression_ratio = compressed_context.compression_ratio

        # Step 6: LLM Response Synthesis
        async with trace_stage("llm_generation", telemetry):
            llm_result = await llm.generate(
                query=request.query,
                context=compressed_context.compressed_text,
                scored_chunks=reranked_chunks,
            )

        # Step 7: Store in Redis Semantic Cache
        async with trace_stage("semantic_cache_store", telemetry):
            await semantic_cache.set(
                tenant_id=tenant_id,
                query_text=request.query,
                query_vector=query_vector,
                response_text=llm_result.answer,
                metadata={"citations": llm_result.citations},
            )

        telemetry.finish(total_start_time)

        sources = [
            SourceDocument(
                doc_id=c.chunk.doc_id,
                content=c.chunk.content,
                score=c.score,
                retrieval_method=c.retrieval_method,
                metadata=c.chunk.metadata,
            )
            for c in reranked_chunks
        ]

        return QueryResponse(
            tenant_id=tenant_id,
            query=request.query,
            answer=llm_result.answer,
            cache_hit=False,
            citations=llm_result.citations,
            sources=sources,
            telemetry=telemetry,
        )

    except HTTPException:
        raise
    except Exception as e:
        telemetry.error = str(e)
        telemetry.finish(total_start_time)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query pipeline error for tenant '{tenant_id}': {str(e)}",
        ) from e


@router.post(
    "/stream",
    summary="Multi-tenant streaming RAG query via Server-Sent Events (SSE)",
)
async def query_pipeline_stream(
    request: QueryRequest,
    tenant_id: str = Depends(get_tenant_id),
    semantic_cache: SemanticCache = Depends(get_semantic_cache),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream token deltas directly to client using SSE."""
    total_start_time = time.perf_counter()
    telemetry = PipelineTelemetry(tenant_id=tenant_id)

    retriever = HybridRetriever(qdrant_client=qdrant, settings=settings)
    reranker = Reranker(settings=settings)
    pruner = ContextPruner(settings=settings)
    llm = LLMClient(settings=settings)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Stage 1: Dense embedding
            yield f"event: stage\ndata: {json.dumps({'stage': 'dense_embedding'})}\n\n"
            query_vector = await retriever.generate_dense_embedding(request.query)

            # Stage 2: Cache lookup
            if not request.bypass_cache:
                yield f"event: stage\ndata: {json.dumps({'stage': 'semantic_cache_lookup'})}\n\n"
                cache_result = await semantic_cache.get(tenant_id, query_vector)
                if cache_result.hit and cache_result.response_text:
                    yield f"event: token\ndata: {json.dumps({'token': cache_result.response_text})}\n\n"
                    telemetry.cache_hit = True
                    telemetry.finish(total_start_time)
                    yield f"event: done\ndata: {json.dumps({'cache_hit': True, 'citations': cache_result.metadata.get('citations', [])})}\n\n"
                    return

            # Stage 3: Hybrid search & Rerank
            yield f"event: stage\ndata: {json.dumps({'stage': 'hybrid_retrieval'})}\n\n"
            fused = await retriever.hybrid_search(
                tenant_id, request.query, query_vector, top_k=settings.hybrid_top_k
            )
            reranked = await reranker.rerank(request.query, fused, top_k=request.top_k)

            # Stage 4: Prune
            yield f"event: stage\ndata: {json.dumps({'stage': 'context_pruning'})}\n\n"
            compressed = await pruner.prune(request.query, reranked)

            # Stage 5: Streaming generation
            yield f"event: stage\ndata: {json.dumps({'stage': 'llm_generation'})}\n\n"
            accumulated_tokens: list[str] = []
            async for token in llm.stream_generate(
                request.query, compressed.compressed_text, reranked
            ):
                accumulated_tokens.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            full_answer = "".join(accumulated_tokens)
            citations = [c.chunk.doc_id for c in reranked]

            # Cache response
            await semantic_cache.set(
                tenant_id=tenant_id,
                query_text=request.query,
                query_vector=query_vector,
                response_text=full_answer,
                metadata={"citations": citations},
            )

            telemetry.finish(total_start_time)
            yield f"event: done\ndata: {json.dumps({'cache_hit': False, 'citations': citations, 'total_ms': telemetry.total_latency_ms})}\n\n"

        except Exception as e:
            logger.error("Error in streaming query pipeline: %s", e)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
