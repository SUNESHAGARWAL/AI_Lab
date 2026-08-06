dev:
    docker compose up -d
    @echo "postgres+redis up. In separate terminals, run:"
    @echo "  uv run uvicorn api.main:app --reload --app-dir apps/api/src"
    @echo "  pnpm --filter web dev"

test:
    uv run pytest -m "not llm and not integration"

evals-fast:
    uv run --package evals python -m evals.cli run-retrieval

evals-full:
    @echo "stub: layers 3+4 (judge model, nightly only) — lands with packages/evals"

ingest:
    uv run --package ingest python -m ingest.cli ingest-corpus

lint:
    uv run ruff check --fix .

typecheck:
    uv run mypy packages/core
