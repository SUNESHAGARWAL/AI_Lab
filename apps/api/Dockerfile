# syntax=docker/dockerfile:1

# Build from the REPO ROOT, not apps/api — this is a uv workspace (root pyproject.toml +
# uv.lock, apps/api/pyproject.toml declares core/llm/retrieval/telemetry as
# `{ workspace = true }` path dependencies), so the build context needs the sibling
# workspace packages. Railway: set the service's build context/root directory to the repo
# root and dockerfilePath to apps/api/Dockerfile (see railway.json at the repo root).
#
# docker build -f apps/api/Dockerfile .

FROM python:3.12-slim-trixie AS builder

# Pinned uv version (matches this repo's dev environment) rather than :latest — per
# CLAUDE.md's "pin exact versions" non-negotiable.
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Deps-only layer, cached independently of source changes — see
# docs.astral.sh/uv/guides/integration/docker/. --no-install-workspace skips the (not yet
# present) workspace member source itself, just resolves third-party dependencies.
#
# Cache mount ids below are hardcoded to this Railway service's id (Railway requires the
# s/<SERVICE_ID>-<name> prefix on every cache mount for its build cache to attach —
# a build ARG doesn't work, since Railway parses the Dockerfile before args are available).
# Harmless for local `docker build` — the id is just a cache namespace token there.
#
# Railway's builder also rejects --mount=type=bind (only type=cache is supported), unlike
# plain BuildKit — so the pyproject.toml files are COPYed in directly instead of
# bind-mounted, ahead of the full source COPY below.
COPY uv.lock pyproject.toml ./
COPY apps/api/pyproject.toml apps/api/pyproject.toml
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY packages/llm/pyproject.toml packages/llm/pyproject.toml
COPY packages/retrieval/pyproject.toml packages/retrieval/pyproject.toml
COPY packages/telemetry/pyproject.toml packages/telemetry/pyproject.toml

RUN --mount=type=cache,id=s/8f040c6b-c0af-4507-8ba7-24ae203fb052-uv-cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --package api

# Now the real source — scoped to exactly what apps/api's dependency closure needs
# (core/llm/retrieval/telemetry per apps/api/pyproject.toml's [tool.uv.sources]), not the
# whole monorepo (apps/ingest and packages/evals pull ragas/deepeval — unwanted weight in
# a production API image).
COPY apps/api apps/api
COPY packages/core packages/core
COPY packages/llm packages/llm
COPY packages/retrieval packages/retrieval
COPY packages/telemetry packages/telemetry

# --frozen, not --locked: --locked re-validates the full workspace's lock consistency,
# which needs every workspace member's pyproject.toml visible (including
# apps/ingest/packages/evals, deliberately excluded from this build context to keep the
# image lean — see the COPY list above). --frozen trusts the already-committed,
# locally-validated uv.lock as-is, which is the property that actually matters for a
# reproducible image build.
#
# Editable (not --no-editable): packages/retrieval's alembic.ini and migrations/ live
# alongside its src/ dir, outside what hatchling packages into the wheel (see
# packages/retrieval/pyproject.toml's [tool.hatch.build.targets.wheel]). migrate.py
# locates alembic.ini via Path(__file__).resolve().parents[2], which only resolves
# correctly when the installed package still points back at its real source tree —
# i.e. an editable install. The runtime stage below COPYs packages/* again so that
# source tree actually exists at runtime, not just during this build.
RUN --mount=type=cache,id=s/8f040c6b-c0af-4507-8ba7-24ae203fb052-uv-cache,target=/root/.cache/uv \
    uv sync --frozen --package api

# packages/retrieval's SentenceTransformerEmbedder runs CPU-only (device="cpu", hardcoded
# — see packages/retrieval/src/retrieval/embedder.py). uv.lock's default resolution pulls
# PyPI's standard torch wheel, which on linux drags in several GB of unused CUDA libraries
# (nvidia-cublas, nvidia-cudnn, cuda-toolkit, triton, ...) purely because they're declared
# metadata dependencies of that wheel — none of it is ever imported or used. Reinstalling
# torch from PyTorch's own CPU-only index, scoped to just this Docker build (not the
# shared workspace lockfile — a workspace-wide `[tool.uv.sources]` index override was
# tried and didn't take effect for this project's multi-platform universal lock, not
# worth the added resolution complexity for a single deploy-time concern), swaps it for
# the ~200MB CPU build with zero CUDA deps.
RUN --mount=type=cache,id=s/8f040c6b-c0af-4507-8ba7-24ae203fb052-uv-cache,target=/root/.cache/uv \
    uv pip install --reinstall --python /app/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

# The swap above uninstalls the CUDA torch package's own metadata but — verified by
# actually building this image and inspecting it — leaves the sibling nvidia-*/triton
# packages' files (~3.5GB) orphaned on disk: `pip list` no longer shows them, but their
# site-packages directories remain. Confirmed importing torch works fully without any
# of this (it's the +cpu build), so it's dead weight, not a real dependency — remove it
# explicitly rather than trust the uninstall to have caught it.
RUN rm -rf /app/.venv/lib/python3.12/site-packages/nvidia \
           /app/.venv/lib/python3.12/site-packages/triton \
           /app/.venv/lib/python3.12/site-packages/nvidia_* \
           /app/.venv/lib/python3.12/site-packages/cuda \
           /app/.venv/lib/python3.12/site-packages/cuda_*

# Bake the embedding model into the image so boot never depends on HuggingFace Hub being
# reachable — see the plan's Stage 1 tradeoff note (reliable network-independent cold start
# over a smaller image). The reranker (core.testing.FakeReranker) needs no weights at all —
# it's an identity passthrough, see docs/adr/0003-reranking-evaluated-and-deferred.md.
ENV HF_HOME=/app/.cache/huggingface
RUN /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"

# --- Runtime stage: only the venv + baked model cache, no uv, no build tooling ---
FROM python:3.12-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 1000 --create-home appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/.cache/huggingface /app/.cache/huggingface
COPY --chown=appuser:appuser apps/api apps/api
COPY --chown=appuser:appuser packages/core packages/core
COPY --chown=appuser:appuser packages/llm packages/llm
COPY --chown=appuser:appuser packages/retrieval packages/retrieval
COPY --chown=appuser:appuser packages/telemetry packages/telemetry

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --loop asyncio, not uvicorn's default uvloop: psycopg_pool's AsyncConnectionPool.open()
# hangs indefinitely under uvloop specifically (reproduced locally — PoolTimeout after the
# configured 10s under uvloop, clean success under plain asyncio). Confirmed via direct
# reproduction outside uvicorn, not a guess.
CMD ["uvicorn", "api.main:app", "--app-dir", "apps/api/src", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
