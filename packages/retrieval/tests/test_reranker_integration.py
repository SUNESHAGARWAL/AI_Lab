import pytest
from core.models import Chunk, Query, ScoredChunk
from retrieval.reranker import CrossEncoderReranker


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_moves_correct_chunk_to_top(reranker: CrossEncoderReranker) -> None:
    # Deliberately mis-ordered: the actually-relevant chunk starts last with the
    # lowest score, as if an upstream retriever ranked it worst.
    candidates = [
        ScoredChunk(
            chunk=Chunk(id="c1", document_id="d", text="Bananas are a good source of potassium."),
            score=0.9,
        ),
        ScoredChunk(
            chunk=Chunk(
                id="c2", document_id="d", text="Photosynthesis converts sunlight into energy."
            ),
            score=0.5,
        ),
        ScoredChunk(
            chunk=Chunk(
                id="c3", document_id="d", text="The Eiffel Tower is located in Paris, France."
            ),
            score=0.1,
        ),
    ]

    reranked = await reranker.rerank(
        Query(text="Where is the Eiffel Tower located?"), candidates
    )

    assert reranked[0].chunk.id == "c3"
    assert reranked[0].score >= reranked[1].score >= reranked[2].score


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_never_drops_or_adds_candidates(reranker: CrossEncoderReranker) -> None:
    candidates = [
        ScoredChunk(
            chunk=Chunk(id=f"c{i}", document_id="d", text=f"Sentence number {i}."), score=0.0
        )
        for i in range(5)
    ]

    reranked = await reranker.rerank(Query(text="a query"), candidates)

    assert len(reranked) == len(candidates)
    assert {sc.chunk.id for sc in reranked} == {sc.chunk.id for sc in candidates}
    assert len({sc.chunk.id for sc in reranked}) == len(reranked)  # no duplicates


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_empty_candidates_returns_empty(reranker: CrossEncoderReranker) -> None:
    assert await reranker.rerank(Query(text="a query"), []) == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_scores_are_normalized(reranker: CrossEncoderReranker) -> None:
    candidates = [
        ScoredChunk(chunk=Chunk(id="c1", document_id="d", text="Some passage text."), score=0.0),
    ]

    reranked = await reranker.rerank(Query(text="a query"), candidates)

    assert 0.0 <= reranked[0].score <= 1.0
