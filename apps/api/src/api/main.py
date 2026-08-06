from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from core.testing import FakeReranker
from fastapi import FastAPI
from redis.asyncio import Redis

from api.config import get_settings
from api.graph import build_graph
from api.graph.checkpointer import postgres_checkpointer
from api.routes.health import router as health_router
from llm import Gateway
from retrieval import PgVectorRetriever, SentenceTransformerEmbedder, apply_migrations, create_pool
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
        # No real reranker adapter yet — an identity-passthrough FakeReranker keeps
        # the graph runnable in the meantime; swap this for a real cross-encoder
        # adapter without touching graph/node code, same as the retriever above.
        reranker = FakeReranker()
        # llm.GatewaySettings reads LLM_REDIS_URL (its own env_prefix), a different
        # var from apps/api's own REDIS_URL — pass our already-validated Redis client
        # explicitly so the app doesn't depend on the two happening to match.
        redis_client = Redis.from_url(settings.redis_url)
        gateway = Gateway(redis_client=redis_client)
        app.state.graph = build_graph(checkpointer, retriever, reranker, gateway)
        logger.info("api.startup", env=settings.app_env)
        try:
            yield
        finally:
            await pool.close()


app = FastAPI(title="api", lifespan=lifespan)
app.include_router(health_router)
