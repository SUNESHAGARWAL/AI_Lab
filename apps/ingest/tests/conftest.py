import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from retrieval import DEFAULT_MODEL_NAME, apply_migrations_sync

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app")
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"


@pytest.fixture(scope="session")
def pg_conn():
    try:
        apply_migrations_sync(DATABASE_URL)
        conn = psycopg.connect(DATABASE_URL)
        register_vector(conn)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(
            f"Postgres not reachable at {DATABASE_URL} ({exc}); run `docker compose up -d`"
        )
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def embedder_model() -> SentenceTransformer:
    return SentenceTransformer(DEFAULT_MODEL_NAME, device="cpu")


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    return CACHE_DIR
