import pytest
from core.models import Chunk, Filters, Query
from core.ports import Embedder, Reranker, Retriever
from core.testing import FakeEmbedder, FakeReranker, FakeRetriever


def _chunk(id: str, document_id: str, text: str, **metadata: str) -> Chunk:
    return Chunk(id=id, document_id=document_id, text=text, metadata=metadata)


def test_fakes_satisfy_their_protocols() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeRetriever(corpus=[]), Retriever)
    assert isinstance(FakeReranker(), Reranker)


@pytest.mark.asyncio
async def test_fake_embedder_is_deterministic_and_shape_correct() -> None:
    embedder = FakeEmbedder()
    query_vector = await embedder.embed_query("hello world")
    again = await embedder.embed_query("hello world")
    assert query_vector == again

    doc_vectors = await embedder.embed_documents(["a", "b", "c"])
    assert len(doc_vectors) == 3
    assert doc_vectors[0] != doc_vectors[1]


@pytest.mark.asyncio
async def test_fake_retriever_orders_by_term_match_and_respects_top_k() -> None:
    corpus = [
        _chunk("1", "doc-a", "the quick brown fox"),
        _chunk("2", "doc-a", "the quick quick fox fox fox"),
        _chunk("3", "doc-b", "completely unrelated text"),
    ]
    retriever = FakeRetriever(corpus=corpus)

    results = await retriever.retrieve(Query(text="quick fox", top_k=2))

    assert len(results) == 2
    assert results[0].chunk.id == "2"
    assert results[0].score >= results[1].score


@pytest.mark.asyncio
async def test_fake_retriever_empty_result_is_not_an_error() -> None:
    retriever = FakeRetriever(corpus=[_chunk("1", "doc-a", "irrelevant")])
    results = await retriever.retrieve(Query(text="nothing matches here"))
    assert results == []


@pytest.mark.asyncio
async def test_fake_retriever_honors_document_id_filter() -> None:
    corpus = [
        _chunk("1", "doc-a", "quick fox"),
        _chunk("2", "doc-b", "quick fox"),
    ]
    retriever = FakeRetriever(corpus=corpus)

    results = await retriever.retrieve(
        Query(text="quick fox", filters=Filters(document_ids=["doc-b"]))
    )

    assert [r.chunk.id for r in results] == ["2"]


@pytest.mark.asyncio
async def test_fake_reranker_identity_preserves_set_and_order_by_default() -> None:
    from core.models import ScoredChunk

    candidates = [
        ScoredChunk(chunk=_chunk("1", "doc-a", "a"), score=0.1),
        ScoredChunk(chunk=_chunk("2", "doc-a", "b"), score=0.9),
    ]
    reranker = FakeReranker()

    result = await reranker.rerank(Query(text="q"), candidates)

    assert result == candidates


@pytest.mark.asyncio
async def test_fake_reranker_score_fn_reorders_without_dropping_candidates() -> None:
    from core.models import ScoredChunk

    candidates = [
        ScoredChunk(chunk=_chunk("1", "doc-a", "a"), score=0.9),
        ScoredChunk(chunk=_chunk("2", "doc-a", "b"), score=0.1),
    ]
    reranker = FakeReranker(score_fn=lambda _query, chunk: 1.0 if chunk.id == "2" else 0.0)

    result = await reranker.rerank(Query(text="q"), candidates)

    assert len(result) == len(candidates)
    assert {sc.chunk.id for sc in result} == {"1", "2"}
    assert result[0].chunk.id == "2"
