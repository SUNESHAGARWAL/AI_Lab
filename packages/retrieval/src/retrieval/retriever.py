from typing import Any

from core.models import Chunk, Filters, Query, ScoredChunk
from core.ports import Embedder
from pgvector import Vector
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def _build_where_clause(filters: Filters | None) -> tuple[str, dict[str, Any]]:
    """Translates a Filters object into a SQL WHERE clause + bound params — filtering
    happens in the query itself, never fetch-then-filter in Python."""
    if filters is None:
        return "TRUE", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}

    if filters.document_ids is not None:
        clauses.append("document_id = ANY(%(document_ids)s)")
        params["document_ids"] = filters.document_ids

    if filters.tags is not None:
        # Matches core.testing.FakeRetriever's tag semantics exactly: metadata["tags"]
        # is a comma-separated string, intersected with the filter's tags — same
        # conceptual filter behaves identically across the fake and this real adapter.
        clauses.append("string_to_array(metadata->>'tags', ',') && %(tags)s")
        params["tags"] = filters.tags

    if filters.metadata_equals is not None:
        for i, (key, value) in enumerate(filters.metadata_equals.items()):
            key_param, value_param = f"meta_key_{i}", f"meta_value_{i}"
            clauses.append(f"metadata ->> %({key_param})s = %({value_param})s")
            params[key_param] = key
            params[value_param] = value

    if not clauses:
        return "TRUE", {}
    return " AND ".join(clauses), params


class PgVectorRetriever:
    """core.ports.Retriever backed by pgvector cosine similarity search — see ADR
    0002. Vector-only: hybrid vector+full-text-search+RRF fusion is future work, not
    needed to satisfy the Retriever port's contract today."""

    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder) -> None:
        self._pool = pool
        self._embedder = embedder

    async def retrieve(self, query: Query) -> list[ScoredChunk]:
        query_vector = await self._embedder.embed_query(query.text)
        where_sql, where_params = _build_where_clause(query.filters)

        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(
                f"""
                SELECT chunk_id, document_id, content, metadata,
                       1 - (embedding <=> %(query_vector)s) AS score
                FROM chunks
                WHERE {where_sql}
                ORDER BY embedding <=> %(query_vector)s
                LIMIT %(top_k)s
                """,
                # pgvector-python only adapts its own Vector wrapper (or numpy arrays)
                # to the `vector` SQL type — a bare Python list falls through to
                # psycopg's default array adapter (double precision[]), which the
                # <=> operator doesn't accept. Confirmed by reading pgvector.psycopg's
                # dumper registration directly after a live "operator does not exist:
                # vector <=> double precision[]" failure.
                {
                    "query_vector": Vector(query_vector),
                    "top_k": query.top_k,
                    **where_params,
                },
            )
            rows = await cur.fetchall()

        return [
            ScoredChunk(
                chunk=Chunk(
                    id=row["chunk_id"],
                    document_id=row["document_id"],
                    text=row["content"],
                    metadata=row["metadata"],
                ),
                score=row["score"],
            )
            for row in rows
        ]
