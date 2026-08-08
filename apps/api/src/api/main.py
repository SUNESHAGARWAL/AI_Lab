from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from core.testing import FakeReranker
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from api.config import get_settings
from api.graph import build_graph
from api.graph.checkpointer import postgres_checkpointer
from api.ratelimit import RateLimiter
from api.routes.health import router as health_router
from api.routes.stream import router as stream_router
from llm import Gateway
from retrieval import (
    PgVectorRetriever,
    SentenceTransformerEmbedder,
    apply_migrations,
    create_pool,
)
from telemetry import get_logger

settings = get_settings()  # fail loudly here, not on first request, if config is missing
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with postgres_checkpointer(settings.database_url) as checkpointer:
        # Must run before the vector-aware pool opens — its configure callback
        # registers the pgvector type adapter, which needs the extension this
        # creates to already exist. See retrieval.migrate's module docstring.
        await apply_migrations(settings.database_url)
        pool = create_pool(settings.database_url)
        await pool.open(wait=True, timeout=10)
        embedder = SentenceTransformerEmbedder()
        retriever = PgVectorRetriever(pool, embedder)
        # Reranking was built (packages/retrieval/src/retrieval/reranker.py) and
        # evaluated (`uv run --package evals python -m evals.cli run-retrieval
        # --rerank`) — BAAI/bge-reranker-base degraded every metric on the real
        # golden set (see docs/adr/0003-reranking-evaluated-and-deferred.md).
        # Deferred until a domain-tuned reranker can be evaluated; identity
        # passthrough keeps the graph's raw PgVectorRetriever cosine order intact.
        reranker = FakeReranker()
        # llm.GatewaySettings reads LLM_REDIS_URL (its own env_prefix), a different
        # var from apps/api's own REDIS_URL — pass our already-validated Redis client
        # explicitly so the app doesn't depend on the two happening to match.
        redis_client = Redis.from_url(settings.redis_url)
        gateway = Gateway(redis_client=redis_client)
        # Same redis_client the gateway/budget guard use — one Redis, no reason to
        # open a second connection just for rate-limit counters.
        rate_limiter = RateLimiter(redis_client, settings)
        # Stashed individually (not just the compiled graph) so api.routes.stream can
        # build a fresh per-request graph wired with a per-request RecordingGateway —
        # see api.graph.streaming.RecordingGateway's docstring for why that isolation
        # matters under concurrent SSE requests.
        app.state.checkpointer = checkpointer
        app.state.retriever = retriever
        app.state.reranker = reranker
        app.state.gateway = gateway
        app.state.rate_limiter = rate_limiter
        app.state.graph = build_graph(checkpointer, retriever, reranker, gateway)
        logger.info("api.startup", env=settings.app_env)
        try:
            yield
        finally:
            await pool.close()


app = FastAPI(title="api", lifespan=lifespan)
# Explicit origin allowlist, never a wildcard — settings.frontend_origin is
# comma-separated so both the deployed Vercel domain and a local dev frontend can be
# listed at once. No credentials (cookies/auth headers) cross this boundary, so
# allow_credentials stays at its default False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.frontend_origin.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(health_router)
app.include_router(stream_router)
