dev:
    docker compose up -d
    @echo "postgres+redis up. In separate terminals, run:"
    @echo "  uv run uvicorn api.main:app --reload --app-dir apps/api/src"
    @echo "  pnpm --filter web dev"

test:
    uv run pytest -m "not llm and not integration"

evals-fast:
    uv run --package evals python -m evals.cli run-retrieval

evals-generation:
    uv run --package evals python -m evals.cli run-generation-eval

evals-judge-agreement:
    uv run --package evals python -m evals.cli judge-agreement-report

evals-full:
    @echo "layer 3 (generation quality, judge model): just evals-generation"
    @echo "read the judge-agreement report first: just evals-judge-agreement"
    @echo "layer 4: not yet built"

ingest:
    uv run --package ingest python -m ingest.cli ingest-corpus

lint:
    uv run ruff check --fix .

typecheck:
    uv run mypy packages/core
