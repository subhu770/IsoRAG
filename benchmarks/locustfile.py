"""Locust Load Test Suite for IsoRAG P95 Latency and Throughput Benchmarking.

Evaluates:
- Cold query latency (Cache miss -> Hybrid search -> Rerank -> LLM generation)
- Warm query latency (Sub-millisecond Redis RediSearch semantic cache hits)
- Multi-tenant concurrency with distinct X-Tenant-ID headers
"""

import random

from locust import HttpUser, between, task

SAMPLE_QUERIES = [
    "What is the company policy on remote work?",
    "How does the quarterly revenue distribution work?",
    "Explain the procedure for disaster recovery and database backups.",
    "What are the compliance requirements for SOC2 type II?",
    "How do I submit an expense report for international travel?",
    "What is the SLA for critical production incidents?",
    "Describe the onboarding checklist for new infrastructure engineers.",
    "What cryptographic standards are required for data at rest?",
]

TENANTS = [
    "tenant-alpha",
    "tenant-beta",
    "tenant-gamma",
    "tenant-delta",
    "tenant-enterprise",
]


class IsoRAGUser(HttpUser):
    """Simulates an enterprise tenant executing queries and ingestion against IsoRAG."""

    wait_time = between(0.1, 1.0)

    def on_start(self) -> None:
        """Assign random tenant partition for each simulated user session."""
        self.tenant_id = random.choice(TENANTS)
        self.headers = {
            "X-Tenant-ID": self.tenant_id,
            "Content-Type": "application/json",
        }

    @task(6)
    def query_with_cache_potential(self) -> None:
        """Simulate repeated and semantically similar queries to measure cache hit P95 latency."""
        query = random.choice(SAMPLE_QUERIES)
        payload = {
            "query": query,
            "top_k": 5,
            "bypass_cache": False,
        }

        with self.client.post(
            "/api/v1/query",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/query [hybrid/cache]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("cache_hit"):
                    response.success()
                else:
                    response.success()
            else:
                response.failure(
                    f"Query failed with status {response.status_code}: {response.text}"
                )

    @task(2)
    def query_bypass_cache(self) -> None:
        """Force cold retrieval pipeline execution to measure P95 full-pipeline latency."""
        query = random.choice(SAMPLE_QUERIES) + f" (req-{random.randint(1000, 9999)})"
        payload = {
            "query": query,
            "top_k": 3,
            "bypass_cache": True,
        }

        with self.client.post(
            "/api/v1/query",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/query [cold-bypass]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Cold query failed with status {response.status_code}: {response.text}"
                )

    @task(1)
    def ingest_batch_documents(self) -> None:
        """Measure ingestion throughput and latency under tenant isolation."""
        doc_count = random.randint(2, 5)
        documents = [
            {
                "doc_id": f"bench-doc-{random.randint(10000, 99999)}",
                "content": f"Tenant {self.tenant_id} operational doc #{i}: {random.choice(SAMPLE_QUERIES)}.",
                "metadata": {"bench": True, "iteration": i},
            }
            for i in range(doc_count)
        ]

        payload = {"documents": documents}

        with self.client.post(
            "/api/v1/ingest",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/ingest [batch]",
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(
                    f"Ingest failed with status {response.status_code}: {response.text}"
                )
