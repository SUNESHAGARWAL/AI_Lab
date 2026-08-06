import pytest
from core.models import Chunk, Query, ScoredChunk
from evals.cli import build_retrieve_fn


class _SpyRetriever:
    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks
        self.last_query: Query | None = None

    async def retrieve(self, query: Query) -> list[ScoredChunk]:
        self.last_query = query
        return self._chunks[: query.top_k]


class _SpyReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: Query, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        self.calls += 1
        return list(reversed(candidates))


def _chunks(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=Chunk(id=f"c{i}", document_id="d", text=f"text {i}"), score=1.0 - i * 0.1)
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_non_rerank_path_never_calls_reranker() -> None:
    retriever = _SpyRetriever(_chunks(10))
    fn = build_retrieve_fn(retriever, None, rerank_pool_size=20)

    ids = await fn("a question", 5)

    assert ids == ["c0", "c1", "c2", "c3", "c4"]  # raw retriever order, unmodified


@pytest.mark.asyncio
async def test_rerank_path_calls_reranker_and_returns_its_order() -> None:
    retriever = _SpyRetriever(_chunks(30))
    reranker = _SpyReranker()
    fn = build_retrieve_fn(retriever, reranker, rerank_pool_size=20)

    ids = await fn("a question", 5)

    assert reranker.calls == 1
    # pool of 20 (c0..c19) fetched, reranker reverses it, top 5 of that is c19..c15
    assert ids == [f"c{i}" for i in range(19, 14, -1)]


@pytest.mark.asyncio
async def test_rerank_path_widens_pool_beyond_top_k() -> None:
    retriever = _SpyRetriever(_chunks(50))
    reranker = _SpyReranker()
    fn = build_retrieve_fn(retriever, reranker, rerank_pool_size=20)

    await fn("a question", 3)

    assert retriever.last_query is not None
    assert retriever.last_query.top_k == 20  # widened pool, not the bare top_k=3


@pytest.mark.asyncio
async def test_rerank_pool_size_never_shrinks_below_requested_top_k() -> None:
    retriever = _SpyRetriever(_chunks(50))
    reranker = _SpyReranker()
    fn = build_retrieve_fn(retriever, reranker, rerank_pool_size=20)

    await fn("a question", 30)

    assert retriever.last_query is not None
    assert retriever.last_query.top_k == 30  # max(pool_size, top_k), pool never shrinks
