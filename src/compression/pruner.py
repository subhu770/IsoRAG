"""Context pruning and token compression using LLMLingua-2 with lightweight fallback."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.core.config import Settings, get_settings
from src.retrieval.hybrid_retriever import ScoredChunk

logger = logging.getLogger("isorag.compression.pruner")


class CompressedContext(BaseModel):
    """Result of context pruning with compression analytics."""

    compressed_text: str = Field(description="Pruned context string ready for prompt injection")
    original_tokens: int = Field(default=0, description="Estimated input token count")
    compressed_tokens: int = Field(default=0, description="Estimated compressed token count")
    compression_ratio: float = Field(
        default=1.0, description="Ratio of compressed to original tokens"
    )
    chunks_used: int = Field(default=0, description="Number of source chunks included")


class ContextPruner:
    """Compresses retrieved context using LLMLingua-2 or an efficient perplexity heuristic."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.enabled = self.settings.enable_context_pruning
        self.target_ratio = self.settings.pruner_target_token_ratio
        self.model_name = self.settings.pruner_model_name
        self._compressor = None

    def _get_compressor(self) -> Any:
        """Lazy load LLMLingua-2 PromptCompressor."""
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor

                logger.info("Initializing LLMLingua-2 compressor with model: %s", self.model_name)
                self._compressor = PromptCompressor(
                    model_name=self.model_name,
                    use_llmlingua2=True,
                )
            except Exception as e:
                logger.warning(
                    "LLMLingua-2 initialization deferred (%s). Using lightweight lexical pruner fallback.",
                    e,
                )
                self._compressor = "fallback"
        return self._compressor

    def _lightweight_prune(self, query: str, context_blocks: list[str], target_ratio: float) -> str:
        """Lightweight sentence-level relevance pruner when neural compressor is offline."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored_sentences = []

        for block in context_blocks:
            sentences = re.split(r"(?<=[.!?])\s+", block)
            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean:
                    continue
                words = re.findall(r"\w+", sent_clean.lower())
                if not words:
                    continue
                overlap = sum(1 for w in words if w in query_terms)
                density = overlap / (len(words) ** 0.5)
                scored_sentences.append((density, sent_clean))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        total_sents = len(scored_sentences)
        keep_count = max(1, int(total_sents * target_ratio))
        selected = [s for _, s in scored_sentences[:keep_count]]
        return "\n\n".join(selected)

    def _compress_sync(self, query: str, chunks: list[ScoredChunk]) -> CompressedContext:
        """Synchronous context compression execution."""
        raw_texts = [f"[{c.chunk.doc_id}] {c.chunk.content}" for c in chunks]
        full_context = "\n\n".join(raw_texts)
        orig_tokens = max(1, len(full_context.split()))

        if not self.enabled or not chunks:
            return CompressedContext(
                compressed_text=full_context,
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                compression_ratio=1.0,
                chunks_used=len(chunks),
            )

        compressor = self._get_compressor()
        if compressor == "fallback" or compressor is None:
            pruned_text = self._lightweight_prune(query, raw_texts, self.target_ratio)
            comp_tokens = max(1, len(pruned_text.split()))
            ratio = round(comp_tokens / orig_tokens, 3)
            return CompressedContext(
                compressed_text=pruned_text,
                original_tokens=orig_tokens,
                compressed_tokens=comp_tokens,
                compression_ratio=ratio,
                chunks_used=len(chunks),
            )

        try:
            res = compressor.compress_prompt(
                context=raw_texts,
                question=query,
                rate=self.target_ratio,
                use_sentence_level_filter=True,
            )
            compressed_prompt = res.get("compressed_prompt", full_context)
            comp_tokens = max(1, len(compressed_prompt.split()))
            ratio = round(comp_tokens / orig_tokens, 3)
            return CompressedContext(
                compressed_text=compressed_prompt,
                original_tokens=orig_tokens,
                compressed_tokens=comp_tokens,
                compression_ratio=ratio,
                chunks_used=len(chunks),
            )
        except Exception as e:
            logger.warning("LLMLingua compression failed: %s. Using uncompressed context.", e)
            return CompressedContext(
                compressed_text=full_context,
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                compression_ratio=1.0,
                chunks_used=len(chunks),
            )

    async def prune(self, query: str, chunks: list[ScoredChunk]) -> CompressedContext:
        """Asynchronously compress retrieved chunks to optimize prompt window and latency."""
        return await asyncio.to_thread(self._compress_sync, query, chunks)
