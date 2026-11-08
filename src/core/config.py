"""Application configuration using Pydantic Settings v2.

Zero hardcoded secrets: all credentials, hosts, and thresholds are configurable
via environment variables with strict type validation.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for IsoRAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --------------------------------------------------------------------------
    # Application Info & Execution Mode
    # --------------------------------------------------------------------------
    app_name: str = Field(default="IsoRAG", description="Application service name")
    app_env: str = Field(
        default="development", description="Environment: development, staging, production"
    )
    debug: bool = Field(default=False, description="Enable debug diagnostics")
    host: str = Field(default="0.0.0.0", description="FastAPI host bind address")
    port: int = Field(default=8000, description="FastAPI port bind address")
    log_level: str = Field(default="INFO", description="Structured log level")

    # --------------------------------------------------------------------------
    # Redis / RediSearch (Tenant Semantic Cache)
    # --------------------------------------------------------------------------
    redis_host: str = Field(default="localhost", description="Redis server hostname")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_password: str | None = Field(default=None, description="Redis authentication password")
    redis_db: int = Field(default=0, description="Redis logical database index")
    redis_ssl: bool = Field(default=False, description="Use SSL/TLS for Redis connection")
    redis_semantic_cache_ttl: int = Field(
        default=86400, description="Semantic cache entry TTL in seconds (default 24h)"
    )
    redis_semantic_cache_index_name: str = Field(
        default="isorag:semantic_cache_idx",
        description="RediSearch index identifier for semantic caching",
    )
    redis_semantic_cache_threshold: float = Field(
        default=0.08,
        description="Cosine distance threshold for semantic cache hit (distance < threshold = hit)",
    )
    redis_semantic_cache_vector_dim: int = Field(
        default=384,
        description="Vector dimension for semantic cache embeddings (matches dense model)",
    )

    # --------------------------------------------------------------------------
    # Qdrant Vector Store (Hard Tenant Isolation via Payload Index)
    # --------------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost", description="Qdrant cluster hostname")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP REST API port")
    qdrant_grpc_port: int = Field(default=6334, description="Qdrant gRPC API port")
    qdrant_url: str | None = Field(
        default=None, description="Qdrant Cloud URL (overrides host/port if set)"
    )
    qdrant_api_key: str | None = Field(
        default=None, description="Qdrant Cloud or protected instance API Key"
    )
    qdrant_collection_name: str = Field(
        default="isorag_documents", description="Primary collection name for multi-tenant storage"
    )
    qdrant_prefer_grpc: bool = Field(
        default=False, description="Use high-throughput gRPC client for Qdrant"
    )
    dense_vector_dim: int = Field(
        default=384, description="Dimensionality of dense embedding vectors"
    )

    # --------------------------------------------------------------------------
    # Embedding Models & Cross-Encoder Reranker
    # --------------------------------------------------------------------------
    dense_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Dense embedding model (HuggingFace/FastEmbed)",
    )
    sparse_model_name: str = Field(
        default="Qdrant/bm25",
        description="Sparse BM25 / SPLADE representation model",
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        description="FlagEmbedding Cross-encoder reranker model",
    )

    # --------------------------------------------------------------------------
    # Retrieval Pipeline Tuning
    # --------------------------------------------------------------------------
    hybrid_top_k: int = Field(
        default=25, description="Number of candidates retrieved per branch before fusion"
    )
    rerank_top_k: int = Field(
        default=5, description="Final number of top candidates selected by reranker"
    )
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing constant k")
    dense_weight: float = Field(default=0.7, description="Dense score weight for hybrid ranking")
    sparse_weight: float = Field(
        default=0.3, description="Sparse BM25 score weight for hybrid ranking"
    )

    # --------------------------------------------------------------------------
    # Context Compression / Token Pruning
    # --------------------------------------------------------------------------
    enable_context_pruning: bool = Field(
        default=True, description="Enable LLMLingua-2 context pruning"
    )
    pruner_target_token_ratio: float = Field(
        default=0.6, description="Target token compression ratio (0.6 = retain 60% of tokens)"
    )
    pruner_model_name: str = Field(
        default="microsoft/llmlingua-2-xlm-roberta-large-meeting",
        description="LLMLingua-2 checkpoint for token perplexity pruning",
    )

    # --------------------------------------------------------------------------
    # LLM Gateway & Simulation
    # --------------------------------------------------------------------------
    simulation_mode: bool = Field(
        default=False,
        description="Enable simulated mock LLM inference mode without external API calls",
    )
    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key (or OpenAI-compatible API key)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible LLM endpoint",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model identifier for synthesis")
    llm_temperature: float = Field(default=0.1, description="Sampling temperature")
    llm_max_tokens: int = Field(default=1024, description="Max response generation tokens")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()
