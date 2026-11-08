"""Batch document ingestion endpoint with hard multi-tenant isolation."""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.api.dependencies import get_qdrant_client, get_tenant_id
from src.api.v1.schemas import IngestRequest, IngestResponse
from src.core.config import Settings, get_settings
from src.retrieval.hybrid_retriever import HybridRetriever

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Async batch ingestion with tenant isolation",
)
async def ingest_documents(
    request: IngestRequest,
    tenant_id: str = Depends(get_tenant_id),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Ingest a batch of documents into Qdrant vector database.

    Enforces hard tenant isolation by embedding `tenant_id` into point payloads
    and ensuring indexed payload filtering.
    """
    start_time = time.perf_counter()
    retriever = HybridRetriever(qdrant_client=qdrant, settings=settings)

    try:
        # Guarantee collection and payload indexes exist
        await retriever.ensure_collection_and_indexes()

        texts = [doc.content for doc in request.documents]

        # Generate dense and sparse embeddings
        dense_model = retriever._get_dense_model()
        sparse_model = retriever._get_sparse_model()

        dense_embeddings = list(dense_model.embed(texts))
        sparse_embeddings = list(sparse_model.embed(texts))

        points: list[qmodels.PointStruct] = []
        for doc, dense_vec, sparse_vec in zip(
            request.documents, dense_embeddings, sparse_embeddings, strict=True
        ):
            # Hard isolation: Stamp tenant_id into point payload
            payload = {
                "tenant_id": tenant_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "ingested_at": time.time(),
            }

            points.append(
                qmodels.PointStruct(
                    id=doc.doc_id,
                    vector={
                        "": dense_vec.tolist(),
                        "bm25": qmodels.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        # Upsert points into Qdrant collection
        await qdrant.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return IngestResponse(
            status="success",
            tenant_id=tenant_id,
            ingested_count=len(points),
            duration_ms=duration_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed for tenant '{tenant_id}': {str(e)}",
        ) from e
