import os

import pytest
import pytest_asyncio
from psycopg_pool import AsyncConnectionPool
from retrieval.embedder import SentenceTransformerEmbedder
from retrieval.migrate import apply_migrations
from retrieval.pool import create_pool

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app")

# The only document_ids this test suite ever seeds. Cleanup is scoped to exactly
# these — never a bare `DELETE FROM chunks` — because this Postgres instance may also
# hold a real ingested corpus (apps/ingest) that these tests must not touch.
TEST_DOCUMENT_IDS = ("doc-a", "doc-b")


@pytest_asyncio.fixture(scope="session")
async def pg_pool():
    try:
        # Must run before the vector-aware pool opens — its configure callback
        # registers the pgvector type adapter, which needs the extension this
        # creates to already exist. See retrieval.migrate's module docstring.
        await apply_migrations(DATABASE_URL)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(
            f"Postgres not reachable at {DATABASE_URL} ({exc}); run `docker compose up -d`"
        )

    pool = create_pool(DATABASE_URL)
    await pool.open(wait=True, timeout=10)
    yield pool
    await pool.close()


@pytest_asyncio.fixture(autouse=True)
async def _clean_chunks_table(pg_pool: AsyncConnectionPool):
    async with pg_pool.connection() as conn:
        await conn.execute(
            "DELETE FROM chunks WHERE document_id = ANY(%s)", (list(TEST_DOCUMENT_IDS),)
        )
    yield
    async with pg_pool.connection() as conn:
        await conn.execute(
            "DELETE FROM chunks WHERE document_id = ANY(%s)", (list(TEST_DOCUMENT_IDS),)
        )


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()
