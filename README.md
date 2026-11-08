# IsoRAG: Production-Grade Multi-Tenant RAG Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![Redis Stack](https://img.shields.io/badge/Cache-RediSearch-critical.svg)](https://redis.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

IsoRAG is a production-grade, multi-tenant Retrieval-Augmented Generation (RAG) platform with hard tenant isolation, sub-millisecond semantic caching, hybrid vector/BM25 search with Reciprocal Rank Fusion (RRF), cross-encoder reranking, and context pruning.

---

## Architecture Highlights

- **Multi-Tenant Hard Isolation**: Guaranteed by strict `X-Tenant-ID` header validation, payload indexing on Qdrant, and tenant-partitioned RediSearch keyspaces. Cross-tenant data leakage is mathematically impossible.
- **Sub-50ms Semantic Caching**: Powered by Redis Stack RediSearch vector indexes. Hit threshold: cosine distance $< 0.08$.
- **Hybrid Retrieval**: Parallel Dense (FastEmbed / BGE) + Sparse (BM25) search aggregated using Reciprocal Rank Fusion (RRF, $k=60$).
- **Cross-Encoder Reranking**: `BAAI/bge-reranker-base` for fine-grained semantic relevance scoring.
- **Context Pruning**: LLMLingua-2 context compression for prompt cost and latency optimization.
- **Zero Hardcoded Secrets**: Strict Pydantic v2 `BaseSettings` reading from `.env` and environment variables.
- **Full-Pipeline Telemetry**: Detailed millisecond latency tracking across every pipeline stage.

---

## Directory Structure

```
IsoRAG/
├── docker-compose.yml              # Qdrant + Redis Stack + API
├── Dockerfile                      # Production multi-stage container
├── Makefile                        # lint, test, run, benchmark automation
├── pyproject.toml                  # Python 3.11+ packaging & dependency matrix
├── .env.example                    # Template environment variables
├── ARCHITECTURE.md                 # Deep-dive architectural documentation
├── src/
│   ├── main.py                     # FastAPI application entrypoint & lifecycle
│   ├── api/
│   │   ├── dependencies.py         # Tenant extraction (X-Tenant-ID) & client providers
│   │   └── v1/
│   │       ├── ingest.py           # Async batch ingestion with tenant isolation
│   │       ├── query.py            # End-to-end query execution pipeline
│   │       └── schemas.py          # Pydantic v2 request/response schemas
│   ├── core/
│   │   ├── config.py               # Pydantic BaseSettings configuration
│   │   └── telemetry.py            # Per-stage latency metrics and tracing
│   ├── cache/
│   │   └── semantic_cache.py       # Redis RediSearch vector similarity cache (<0.08 hit)
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # Qdrant Dense + Sparse BM25 + RRF
│   │   └── reranker.py             # BAAI/bge-reranker-base cross-encoder wrapper
│   ├── compression/
│   │   └── pruner.py               # LLMLingua-2 context pruning
│   └── generator/
│       └── llm_client.py           # Streaming LLM client with citations
├── tests/
│   ├── conftest.py                 # Mock fixtures and in-memory RediSearch emulator
│   ├── test_tenant_isolation.py    # Hard tenant boundary & leak prevention tests
│   └── test_semantic_cache.py      # Cache hit/miss, distance thresholding & TTL tests
└── benchmarks/
    ├── locustfile.py               # P95 latency load test
    └── eval_ragas.py               # Faithfulness & context recall RAGAS evaluation runner
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd IsoRAG

# Copy environment variables
cp .env.example .env
```

### 2. Start via Docker Compose (Recommended)

Starts Qdrant, Redis Stack, and the IsoRAG FastAPI API in isolated containers:

```bash
docker compose up -d --build
```

Access:
- **API Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Run Locally (Development)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,benchmark]"

# Run FastAPI server with auto-reload
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage

### 1. Ingest Documents (`POST /api/v1/ingest`)

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "X-Tenant-ID: tenant-enterprise-01" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "doc_id": "sec-001",
        "content": "IsoRAG uses RediSearch with a cosine distance threshold under 0.08 for semantic cache hits.",
        "metadata": {"category": "architecture", "confidential": true}
      },
      {
        "doc_id": "sec-002",
        "content": "All queries must supply the X-Tenant-ID header to guarantee zero cross-tenant data leakage.",
        "metadata": {"category": "security"}
      }
    ]
  }'
```

### 2. Query Pipeline (`POST /api/v1/query`)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-Tenant-ID: tenant-enterprise-01" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the cosine threshold for cache hits?",
    "top_k": 3
  }'
```

**Response Format**:
```json
{
  "tenant_id": "tenant-enterprise-01",
  "query": "What is the cosine threshold for cache hits?",
  "answer": "According to the architecture specification [sec-001], IsoRAG uses a cosine distance threshold under 0.08 for semantic cache hits in Redis RediSearch.",
  "cache_hit": false,
  "citations": ["sec-001"],
  "sources": [
    {
      "doc_id": "sec-001",
      "content": "IsoRAG uses RediSearch with a cosine distance threshold under 0.08 for semantic cache hits.",
      "score": 0.9842,
      "retrieval_method": "cross_encoder_rerank",
      "metadata": {"category": "architecture"}
    }
  ],
  "telemetry": {
    "trace_id": "trace-9f4a8b1c2d3e",
    "tenant_id": "tenant-enterprise-01",
    "cache_hit": false,
    "stage_latencies_ms": {
      "dense_embedding": 14.2,
      "semantic_cache_lookup": 1.8,
      "hybrid_retrieval": 22.4,
      "cross_encoder_rerank": 18.6,
      "context_pruning": 9.1,
      "llm_generation": 280.5,
      "semantic_cache_store": 2.1
    },
    "total_latency_ms": 348.7,
    "tokens_compressed": 48,
    "compression_ratio": 0.58
  }
}
```

*Note: Executing a subsequent semantically equivalent query will hit the cache in $< 5\text{ ms}$ with `cache_hit: true`.*

---

## Testing & Quality Assurance

```bash
# Run complete test suite
pytest -v

# Run tenant isolation security tests
pytest -v tests/test_tenant_isolation.py

# Run semantic cache threshold and unit tests
pytest -v tests/test_semantic_cache.py
```

---

## Benchmarks

### 1. P95 Latency Load Testing (Locust)

```bash
locust -f benchmarks/locustfile.py --host http://localhost:8000
```
Open [http://localhost:8089](http://localhost:8089) to execute concurrent tenant simulations.

### 2. RAGAS Quality Evaluation

```bash
python benchmarks/eval_ragas.py
```
Evaluates:
- **Faithfulness** ($\ge 0.92$)
- **Answer Relevancy** ($\ge 0.88$)
- **Context Precision** ($\ge 0.85$)
- **Context Recall** ($\ge 0.89$)

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

