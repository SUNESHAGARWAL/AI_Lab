dev:
    docker compose up -d
    @echo "postgres+redis up. In separate terminals, run:"
    @echo "  uv run uvicorn api.main:app --reload --app-dir apps/api/src"
    @echo "  pnpm --filter web dev"

test:
    uv run pytest -m "not llm"

evals-fast:
    @echo "stub: layers 1+2 (deterministic, free) — lands with packages/evals"

evals-full:
    @echo "stub: layers 3+4 (judge model, nightly only) — lands with packages/evals"

ingest:
    uv run --package ingest python -m ingest.cli

lint:
    uv run ruff check --fix .

typecheck:
    uv run mypy packages/core
