import pytest
from core.models import Chunk, Filters, Query
from pgvector import Vector
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from retrieval.embedder import SentenceTransformerEmbedder
from retrieval.retriever import PgVectorRetriever


async def _seed(
    pool: AsyncConnectionPool, embedder: SentenceTransformerEmbedder, chunks: list[Chunk]
) -> None:
    vectors = await embedder.embed_documents([c.text for c in chunks])
    async with pool.connection() as conn:
        for chunk, vector in zip(chunks, vectors, strict=True):
            await conn.execute(
                "INSERT INTO chunks (chunk_id, document_id, content, embedding, metadata) "
                "VALUES (%s, %s, %s, %s, %s)",
                (chunk.id, chunk.document_id, chunk.text, Vector(vector), Jsonb(chunk.metadata)),
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_returns_semantically_relevant_chunk_first(
    pg_pool: AsyncConnectionPool, embedder: SentenceTransformerEmbedder
) -> None:
    chunks = [
        Chunk(id="c1", document_id="doc-a", text="The Eiffel Tower is located in Paris, France."),
        Chunk(id="c2", document_id="doc-a", text="Bananas are a good source of potassium."),
        Chunk(
            id="c3",
            document_id="doc-b",
            text="Photosynthesis converts sunlight into chemical energy in plants.",
        ),
    ]
    await _seed(pg_pool, embedder, chunks)
    retriever = PgVectorRetriever(pg_pool, embedder)

    results = await retriever.retrieve(Query(text="Where is the Eiffel Tower?", top_k=3))

    assert len(results) == 3
    assert results[0].chunk.id == "c1"
    assert results[0].score >= results[1].score >= results[2].score


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_honors_document_id_filter(
    pg_pool: AsyncConnectionPool, embedder: SentenceTransformerEmbedder
) -> None:
    chunks = [
        Chunk(id="c1", document_id="doc-a", text="The Eiffel Tower is located in Paris, France."),
        Chunk(
            id="c2",
            document_id="doc-b",
            text="The Eiffel Tower was completed in 1889 for the World's Fair.",
        ),
    ]
    await _seed(pg_pool, embedder, chunks)
    retriever = PgVectorRetriever(pg_pool, embedder)

    results = await retriever.retrieve(
        Query(text="Tell me about the Eiffel Tower", filters=Filters(document_ids=["doc-b"]))
    )

    assert [r.chunk.id for r in results] == ["c2"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_honors_top_k(
    pg_pool: AsyncConnectionPool, embedder: SentenceTransformerEmbedder
) -> None:
    chunks = [
        Chunk(id=f"c{i}", document_id="doc-a", text=f"Fact number {i} about the Eiffel Tower.")
        for i in range(5)
    ]
    await _seed(pg_pool, embedder, chunks)
    retriever = PgVectorRetriever(pg_pool, embedder)

    results = await retriever.retrieve(Query(text="Eiffel Tower", top_k=2))

    assert len(results) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_no_chunks_match_filter(
    pg_pool: AsyncConnectionPool, embedder: SentenceTransformerEmbedder
) -> None:
    # Not a literally empty table — this Postgres instance may hold a real ingested
    # corpus alongside this suite's own doc-a/doc-b chunks (see conftest.py's scoped
    # cleanup) — so "no match" is proven via a filter with no matching document_id,
    # not via table emptiness.
    retriever = PgVectorRetriever(pg_pool, embedder)

    results = await retriever.retrieve(
        Query(text="anything at all", filters=Filters(document_ids=["doc-a"]))
    )

    assert results == []
