"""Cross-encoder reranking wrapper using FlagEmbedding (bge-reranker-base)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.config import Settings, get_settings
from src.retrieval.hybrid_retriever import ScoredChunk

logger = logging.getLogger("isorag.retrieval.reranker")


class Reranker:
    """Wraps FlagEmbedding FlagReranker for high-precision cross-encoder scoring."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.reranker_model_name
        self.top_k = self.settings.rerank_top_k
        self._reranker = None

    def _get_model(self) -> Any:
        """Lazy load FlagReranker instance."""
        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker

                logger.info("Loading cross-encoder reranker model: %s", self.model_name)
                self._reranker = FlagReranker(self.model_name, use_fp16=True)
            except Exception as e:
                logger.warning(
                    "FlagReranker initialization notice (%s). Falling back to CPU / pass-through mode.",
                    e,
                )
                try:
                    from FlagEmbedding import FlagReranker

                    self._reranker = FlagReranker(self.model_name, use_fp16=False)
                except Exception as ex:
                    logger.error(
                        "Could not load FlagReranker: %s. Using heuristic reranker fallback.", ex
                    )
                    self._reranker = "fallback"
        return self._reranker

    def _score_pairs_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-encoder scores synchronously."""
        model = self._get_model()
        if model == "fallback" or model is None:
            # Fallback: maintain original order with dummy scores
            return [1.0 / (i + 1) for i in range(len(pairs))]

        raw_scores = model.compute_score(pairs)
        if isinstance(raw_scores, (int, float)):
            return [float(raw_scores)]
        return [float(s) for s in raw_scores]

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Rerank retrieval candidates using cross-encoder deep relevance scoring.

        Args:
            query: User's natural language question.
            candidates: Candidates produced by hybrid retrieval.
            top_k: Desired number of top reranked chunks.

        Returns:
            Sorted list of ScoredChunk with updated cross-encoder scores.
        """
        if not candidates:
            return []

        limit = top_k or self.top_k
        pairs = [(query, c.chunk.content) for c in candidates]

        scores = await asyncio.to_thread(self._score_pairs_sync, pairs)

        reranked: list[ScoredChunk] = []
        for candidate, score in zip(candidates, scores, strict=True):
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=round(score, 4),
                    retrieval_method="cross_encoder_rerank",
                )
            )

        # Sort descending by cross-encoder score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:limit]
