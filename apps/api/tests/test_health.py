import os

os.environ.setdefault("DATABASE_URL", "postgresql://app:app@localhost:5432/app")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest
from api.main import app
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "env": "development"}


class _FakePool:
    """Mimics psycopg_pool's connection()/cursor() async context managers."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def connection(self):  # type: ignore[no-untyped-def]
        if self._error:
            raise self._error
        return self

    def cursor(self):  # type: ignore[no-untyped-def]
        return self

    async def __aenter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, *args: object) -> None:
        return None


class _FakeRedis:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def ping(self) -> bool:
        if self._error:
            raise self._error
        return True


async def _ready(pool: object, redis: object) -> tuple[int, dict]:  # type: ignore[type-arg]
    app.state.pool = pool
    app.state.redis_client = redis
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")
    return response.status_code, response.json()


@pytest.mark.asyncio
async def test_ready_reports_ok_when_every_dependency_answers() -> None:
    status, body = await _ready(_FakePool(), _FakeRedis())

    assert status == 200
    assert body == {"status": "ok", "checks": {"postgres": "ok", "redis": "ok"}}


@pytest.mark.asyncio
async def test_ready_names_the_dependency_that_is_down() -> None:
    """The whole point: an unreachable Postgres must be visible from outside. During the
    outage /health said ok while every query failed, which is what made a dead backend
    look like a frontend bug."""
    status, body = await _ready(_FakePool(error=ConnectionError("server closed")), _FakeRedis())

    assert status == 503
    assert body["status"] == "degraded"
    assert body["checks"]["postgres"] == "ConnectionError"
    assert body["checks"]["redis"] == "ok"


@pytest.mark.asyncio
async def test_ready_is_degraded_when_redis_is_unreachable() -> None:
    status, body = await _ready(_FakePool(), _FakeRedis(error=ConnectionError("closed")))

    assert status == 503
    assert body["checks"]["redis"] == "ConnectionError"


@pytest.mark.asyncio
async def test_liveness_stays_dependency_free_even_when_everything_is_down() -> None:
    """Railway restarts the container ON_FAILURE against /health, so a suspended Neon
    must not be able to trigger a restart loop."""
    app.state.pool = _FakePool(error=ConnectionError("server closed"))
    app.state.redis_client = _FakeRedis(error=ConnectionError("closed"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
