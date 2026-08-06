from pathlib import Path

import psycopg
import pytest
from ingest.documents import DocumentSource
from ingest.pipeline import ingest_document
from sentence_transformers import SentenceTransformer

# Same real document (celex_id) as the production "eu_ai_act" source, but a
# dedicated test-only slug — isolated document_id/chunk_id namespace so this test
# never collides with or clobbers a real ingest of the same corpus.
TEST_SOURCE = DocumentSource(
    slug="eu_ai_act_test_sample",
    name="Regulation (EU) 2024/1689 (Artificial Intelligence Act) [test sample]",
    celex_id="32024R1689",
    eli_uri="http://data.europa.eu/eli/reg/2024/1689/oj",
)


@pytest.fixture(autouse=True)
def _clean_test_chunks(pg_conn: psycopg.Connection):
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (TEST_SOURCE.slug,))
    pg_conn.commit()
    yield
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (TEST_SOURCE.slug,))
    pg_conn.commit()


def _chunk_ids(conn: psycopg.Connection, document_id: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM chunks WHERE document_id = %s", (document_id,))
        return {row[0] for row in cur.fetchall()}


@pytest.mark.integration
def test_ingest_document_is_idempotent_on_second_run(
    pg_conn: psycopg.Connection, embedder_model: SentenceTransformer, cache_dir: Path
) -> None:
    stats_first = ingest_document(
        TEST_SOURCE, pg_conn, embedder_model, cache_dir, max_items=2
    )

    assert stats_first.articles == 2
    assert stats_first.recitals == 2
    assert stats_first.inserted == stats_first.chunks
    assert stats_first.chunks > 0

    chunk_ids_first = _chunk_ids(pg_conn, TEST_SOURCE.slug)
    assert len(chunk_ids_first) == stats_first.chunks

    stats_second = ingest_document(
        TEST_SOURCE, pg_conn, embedder_model, cache_dir, max_items=2
    )
    chunk_ids_second = _chunk_ids(pg_conn, TEST_SOURCE.slug)

    assert stats_second.chunks == stats_first.chunks
    assert chunk_ids_second == chunk_ids_first
