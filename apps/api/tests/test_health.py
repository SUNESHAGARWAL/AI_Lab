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
