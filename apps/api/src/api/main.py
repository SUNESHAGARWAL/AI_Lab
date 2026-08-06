from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import get_settings
from api.graph import build_graph
from api.graph.checkpointer import postgres_checkpointer
from api.routes.health import router as health_router
from telemetry import get_logger

settings = get_settings()  # fail loudly here, not on first request, if config is missing
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with postgres_checkpointer(settings.database_url) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        logger.info("api.startup", env=settings.app_env)
        yield


app = FastAPI(title="api", lifespan=lifespan)
app.include_router(health_router)
