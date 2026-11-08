.PHONY: help install dev lint format test test-unit test-tenant run docker-up docker-down benchmark-locust benchmark-ragas clean

PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn

help:
	@echo "IsoRAG Staff Infrastructure Automation Commands:"
	@echo "  make install           Install base production dependencies"
	@echo "  make dev               Install development & benchmark dependencies"
	@echo "  make lint              Run ruff linter and mypy type checks"
	@echo "  make format            Run code formatters (ruff format)"
	@echo "  make test              Run all pytest test suites"
	@echo "  make test-unit         Run unit test suites"
	@echo "  make test-tenant       Verify hard tenant isolation tests"
	@echo "  make run               Run FastAPI server locally with reload"
	@echo "  make docker-up         Start Qdrant, Redis Stack, and IsoRAG API"
	@echo "  make docker-down       Stop and purge docker containers"
	@echo "  make benchmark-locust  Run Locust load tests against IsoRAG query endpoint"
	@echo "  make benchmark-ragas   Run RAGAS evaluation runner"
	@echo "  make clean             Clean temporary files and caches"

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev,benchmark]"

lint:
	ruff check src tests benchmarks
	mypy src

format:
	ruff format src tests benchmarks
	ruff check --fix src tests benchmarks

test:
	pytest -v --cov=src --cov-report=term-missing tests/

test-unit:
	pytest -v tests/test_semantic_cache.py

test-tenant:
	pytest -v tests/test_tenant_isolation.py

run:
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

benchmark-locust:
	locust -f benchmarks/locustfile.py --host http://localhost:8000

benchmark-ragas:
	$(PYTHON) benchmarks/eval_ragas.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache

