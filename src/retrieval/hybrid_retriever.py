"""Hybrid retrieval engine combining Dense Qdrant vectors and Sparse BM25 with RRF.

Key Architectural Guarantees:
- Hard Tenant Isolation: Every Qdrant query enforces a strict payload filter on `tenant_id`.
- Payload Indexing: Verifies keyword payload index exists on `tenant_id` for sub-linear search time.
- Reciprocal Rank Fusion (RRF): Blends dense vector semantics with exact lexical keyword matching.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import Settings, get_settings

logger = logging.getLogger("isorag.retrieval.hybrid")


class DocumentChunk(BaseModel):
    """Normalized document chunk for ingestion and retrieval."""

    doc_id: str
    tenant_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """Document chunk annotated with relevance score and retrieval provenance."""

    chunk: DocumentChunk
    score: float
    retrieval_method: str = "hybrid_rrf"


class HybridRetriever:
    """Orchestrates multi-tenant hybrid search across Qdrant and Sparse BM25."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        settings: Settings | None = None,
    ) -> None:
        self.qdrant = qdrant_client
        self.settings = settings or get_settings()
        self.collection_name = self.settings.qdrant_collection_name
        self.dense_dim = self.settings.dense_vector_dim
        self.rrf_k = self.settings.rrf_k
        self._embedding_model = None
        self._sparse_model = None
        self._collection_initialized = False

    def _get_dense_model(self) -> Any:
        """Lazy load dense FastEmbed embedding model."""
        if self._embedding_model is None:
            from fastembed import TextEmbedding

            self._embedding_model = TextEmbedding(model_name=self.settings.dense_model_name)
        return self._embedding_model

    def _get_sparse_model(self) -> Any:
        """Lazy load sparse FastEmbed BM25 model."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding

            self._sparse_model = SparseTextEmbedding(model_name=self.settings.sparse_model_name)
        return self._sparse_model

    async def ensure_collection_and_indexes(self) -> None:
        """Ensure Qdrant collection and tenant payload indexes are created."""
        if self._collection_initialized:
            return

        try:
            collections = await self.qdrant.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self.collection_name not in existing_names:
                await self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.dense_dim,
                        distance=qmodels.Distance.COSINE,
                    ),
                    sparse_vectors_config={
                        "bm25": qmodels.SparseVectorParams(
                            modifier=qmodels.Modifier.IDF,
                        )
                    },
                )
                logger.info("Created Qdrant collection: %s", self.collection_name)

            # Enforce keyword payload indexing on tenant_id for hard tenant isolation
            await self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            logger.info("Ensured payload index on 'tenant_id' in Qdrant.")
            self._collection_initialized = True
        except Exception as e:
            logger.warning("Qdrant collection setup notice: %s", e)
            self._collection_initialized = True

    async def generate_dense_embedding(self, text: str) -> list[float]:
        """Generate dense embedding vector for query or document."""

        def _embed() -> list[float]:
            model = self._get_dense_model()
            embeddings = list(model.embed([text]))
            return embeddings[0].tolist()

        return await asyncio.to_thread(_embed)

    async def generate_sparse_embedding(self, text: str) -> tuple[list[int], list[float]]:
        """Generate sparse BM25 indices and values for query or document."""

        def _sparse_embed() -> tuple[list[int], list[float]]:
            model = self._get_sparse_model()
            sparse_vectors = list(model.embed([text]))
            sv = sparse_vectors[0]
            return sv.indices.tolist(), sv.values.tolist()

        return await asyncio.to_thread(_sparse_embed)

    async def retrieve_dense(
        self,
        tenant_id: str,
        query_vector: list[float],
        limit: int,
    ) -> list[ScoredChunk]:
        """Execute dense vector search strictly filtered by tenant_id."""
        tenant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="tenant_id",
                    match=qmodels.MatchValue(value=tenant_id),
                )
            ]
        )

        if hasattr(self.qdrant, "query_points"):
            resp = await self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=tenant_filter,
                limit=limit,
                with_payload=True,
            )
            hits = getattr(resp, "points", resp)
        else:
            hits = await self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=tenant_filter,
                limit=limit,
                with_payload=True,
            )

        results: list[ScoredChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = DocumentChunk(
                doc_id=str(hit.id),
                tenant_id=payload.get("tenant_id", tenant_id),
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
            )
            results.append(ScoredChunk(chunk=chunk, score=hit.score, retrieval_method="dense"))
        return results

    async def retrieve_sparse(
        self,
        tenant_id: str,
        query_sparse: tuple[list[int], list[float]],
        limit: int,
    ) -> list[ScoredChunk]:
        """Execute sparse BM25 search strictly filtered by tenant_id."""
        indices, values = query_sparse
        tenant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="tenant_id",
                    match=qmodels.MatchValue(value=tenant_id),
                )
            ]
        )

        sparse_vector = qmodels.SparseVector(indices=indices, values=values)
        if hasattr(self.qdrant, "query_points"):
            resp = await self.qdrant.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using="bm25",
                query_filter=tenant_filter,
                limit=limit,
                with_payload=True,
            )
            hits = getattr(resp, "points", resp)
        else:
            hits = await self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=sparse_vector,
                using="bm25",
                query_filter=tenant_filter,
                limit=limit,
                with_payload=True,
            )

        results: list[ScoredChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = DocumentChunk(
                doc_id=str(hit.id),
                tenant_id=payload.get("tenant_id", tenant_id),
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
            )
            results.append(ScoredChunk(chunk=chunk, score=hit.score, retrieval_method="sparse"))
        return results

    def reciprocal_rank_fusion(
        self,
        dense_results: list[ScoredChunk],
        sparse_results: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Blend dense and sparse ranking using Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_score(d) = dense_weight / (k + rank_dense(d)) + sparse_weight / (k + rank_sparse(d))
        """
        dense_weight = self.settings.dense_weight
        sparse_weight = self.settings.sparse_weight
        k = self.rrf_k

        doc_scores: dict[str, float] = {}
        chunks_map: dict[str, DocumentChunk] = {}

        # Process dense ranking
        for rank, item in enumerate(dense_results, start=1):
            doc_id = item.chunk.doc_id
            chunks_map[doc_id] = item.chunk
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (dense_weight / (k + rank))

        # Process sparse ranking
        for rank, item in enumerate(sparse_results, start=1):
            doc_id = item.chunk.doc_id
            chunks_map[doc_id] = item.chunk
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (sparse_weight / (k + rank))

        # Sort combined results descending by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        return [
            ScoredChunk(
                chunk=chunks_map[doc_id],
                score=round(score, 6),
                retrieval_method="hybrid_rrf",
            )
            for doc_id, score in sorted_docs
        ]

    async def hybrid_search(
        self,
        tenant_id: str,
        query: str,
        query_dense: list[float],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Execute concurrent dense and sparse searches with RRF aggregation."""
        await self.ensure_collection_and_indexes()
        limit = top_k or self.settings.hybrid_top_k

        # Generate sparse vector concurrently
        sparse_indices_values = await self.generate_sparse_embedding(query)

        # Run dense and sparse retrieval in parallel with tenant filter
        dense_task = self.retrieve_dense(tenant_id, query_dense, limit)
        sparse_task = self.retrieve_sparse(tenant_id, sparse_indices_values, limit)

        dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task)

        # Apply Reciprocal Rank Fusion
        fused = self.reciprocal_rank_fusion(dense_res, sparse_res, limit)
        return fused
