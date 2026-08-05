from fastapi import FastAPI

from api.config import get_settings
from api.routes.health import router as health_router
from telemetry import get_logger

settings = get_settings()  # fail loudly here, not on first request, if config is missing
logger = get_logger(__name__)

app = FastAPI(title="api")
app.include_router(health_router)

logger.info("api.startup", env=settings.app_env)
