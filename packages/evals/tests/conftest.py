import os

import pytest

from retrieval import apply_migrations, create_pool

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app")


@pytest.fixture(scope="session")
def database_url() -> str:
    return DATABASE_URL


@pytest.fixture(scope="session")
def require_postgres() -> None:
    # NOT autouse — only test_cli_integration.py (the @pytest.mark.integration tests)
    # should pay for a live-Postgres check; the pure metric/schema/scorecard unit
    # tests in this same directory must stay network-free.
    import asyncio

    async def _check() -> None:
        await apply_migrations(DATABASE_URL)
        pool = create_pool(DATABASE_URL)
        await pool.open(wait=True, timeout=10)
        await pool.close()

    try:
        asyncio.run(_check())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(
            f"Postgres not reachable at {DATABASE_URL} ({exc}); run `docker compose up -d`"
        )
