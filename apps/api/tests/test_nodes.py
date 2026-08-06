import json
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from api.graph.nodes import (
    make_critic_node,
    make_generator_node,
    make_reranker_node,
    make_retriever_node,
    route_after_critic,
)
from api.graph.state import initial_state
from core.models import Chunk, Filters, ScoredChunk
from core.testing import FakeReranker, FakeRetriever
from fakeredis import FakeAsyncRedis
from llm.config import GatewaySettings

from llm import AllProvidersExhausted, Gateway


def _chunk(id: str, document_id: str, text: str) -> Chunk:
    return Chunk(id=id, document_id=document_id, text=text)


@pytest.mark.asyncio
async def test_retriever_node_writes_scored_chunks_to_state() -> None:
    corpus = [
        _chunk("1", "doc-a", "the quick brown fox"),
        _chunk("2", "doc-a", "the quick quick fox fox fox"),
        _chunk("3", "doc-b", "completely unrelated text"),
    ]
    node = make_retriever_node(FakeRetriever(corpus=corpus))
    state = initial_state("quick fox")

    result = await node(state)

    chunks = result["retrieved_chunks"]
    assert [sc.chunk.id for sc in chunks] == ["2", "1"]
    assert chunks[0].score >= chunks[1].score


@pytest.mark.asyncio
async def test_retriever_node_prefers_rewritten_query_over_raw_query() -> None:
    corpus = [
        _chunk("1", "doc-a", "raw query terms"),
        _chunk("2", "doc-a", "rewritten query terms"),
    ]
    node = make_retriever_node(FakeRetriever(corpus=corpus))
    state = initial_state("raw")
    state["rewritten_query"] = "rewritten"

    result = await node(state)

    assert [sc.chunk.id for sc in result["retrieved_chunks"]] == ["2"]


@pytest.mark.asyncio
async def test_retriever_node_honors_filters_from_state() -> None:
    corpus = [
        _chunk("1", "doc-a", "quick fox"),
        _chunk("2", "doc-b", "quick fox"),
    ]
    node = make_retriever_node(FakeRetriever(corpus=corpus))
    state = initial_state("quick fox", filters=Filters(document_ids=["doc-b"]))

    result = await node(state)

    assert [sc.chunk.id for sc in result["retrieved_chunks"]] == ["2"]


@pytest.mark.asyncio
async def test_retriever_node_returns_empty_list_without_error_on_no_match() -> None:
    node = make_retriever_node(FakeRetriever(corpus=[_chunk("1", "doc-a", "irrelevant")]))
    state = initial_state("nothing matches here")

    result = await node(state)

    assert result["retrieved_chunks"] == []


def _scored(id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(id, "doc-a", f"chunk {id}"), score=score)


@pytest.mark.asyncio
async def test_reranker_node_truncates_to_top_n() -> None:
    candidates = [
        _scored("1", 0.1),
        _scored("2", 0.9),
        _scored("3", 0.5),
        _scored("4", 0.7),
        _scored("5", 0.3),
    ]
    score_by_id = {sc.chunk.id: sc.score for sc in candidates}
    reranker = FakeReranker(score_fn=lambda _query, chunk: score_by_id[chunk.id])
    node = make_reranker_node(reranker, top_n=2)
    state = initial_state("q")
    state["retrieved_chunks"] = candidates

    result = await node(state)

    assert [sc.chunk.id for sc in result["reranked_chunks"]] == ["2", "4"]


@pytest.mark.asyncio
async def test_reranker_node_honors_identity_default() -> None:
    candidates = [_scored("1", 0.5), _scored("2", 0.9)]
    node = make_reranker_node(FakeReranker(), top_n=5)
    state = initial_state("q")
    state["retrieved_chunks"] = candidates

    result = await node(state)

    assert result["reranked_chunks"] == candidates


@pytest.mark.asyncio
async def test_reranker_node_honors_custom_score_fn_ordering() -> None:
    candidates = [_scored("1", 0.9), _scored("2", 0.1)]
    # score_fn inverts the original ranking entirely
    reranker = FakeReranker(score_fn=lambda _query, chunk: 1.0 if chunk.id == "2" else 0.0)
    node = make_reranker_node(reranker, top_n=5)
    state = initial_state("q")
    state["retrieved_chunks"] = candidates

    result = await node(state)

    assert [sc.chunk.id for sc in result["reranked_chunks"]] == ["2", "1"]


@pytest.mark.asyncio
async def test_reranker_node_empty_input_returns_empty_list() -> None:
    node = make_reranker_node(FakeReranker())
    state = initial_state("q")
    state["retrieved_chunks"] = []

    result = await node(state)

    assert result["reranked_chunks"] == []


# --- generator node -----------------------------------------------------------
# Local copies of packages/llm/tests/conftest.py's make_response/make_completion_fn
# pattern (SimpleNamespace-shaped fake litellm.ModelResponse, no real LiteLLM class
# involved) — not importable across package test dirs, so duplicated here.


def _completion_response(
    text: str, prompt_tokens: int = 10, completion_tokens: int = 5
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_completion_fn(
    text: str,
) -> tuple[Callable[..., Awaitable[SimpleNamespace]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    async def _fn(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion_response(text)

    return _fn, calls


def _gateway(
    completion_fn: Callable[..., Awaitable[SimpleNamespace]], **settings_kwargs: Any
) -> Gateway:
    return Gateway(
        settings=GatewaySettings(**settings_kwargs),
        redis_client=FakeAsyncRedis(),
        completion_fn=completion_fn,
    )


@pytest.mark.asyncio
async def test_generator_node_passes_through_valid_citations() -> None:
    chunks = [_scored("1", 0.9), _scored("2", 0.5)]
    payload = json.dumps(
        {
            "answer": "The answer is X.",
            "citations": ["1"],
            "confidence": 0.8,
            "abstained": False,
        }
    )
    completion_fn, calls = _make_completion_fn(payload)
    node = make_generator_node(_gateway(completion_fn))
    state = initial_state("q")
    state["reranked_chunks"] = chunks

    result = await node(state)

    assert result["citations"] == ["1"]
    assert result["abstained"] is False
    assert result["answer"] == "The answer is X."
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_generator_node_forces_abstention_on_unknown_citation() -> None:
    chunks = [_scored("1", 0.9)]
    payload = json.dumps(
        {
            "answer": "The answer is X.",
            "citations": ["99"],  # not a real input chunk id
            "confidence": 0.9,
            "abstained": False,
        }
    )
    completion_fn, _ = _make_completion_fn(payload)
    node = make_generator_node(_gateway(completion_fn))
    state = initial_state("q")
    state["reranked_chunks"] = chunks

    result = await node(state)

    assert result["abstained"] is True
    assert "99" in (result["abstain_reason"] or "")


@pytest.mark.asyncio
async def test_generator_node_abstains_on_empty_chunks_without_calling_gateway() -> None:
    completion_fn, calls = _make_completion_fn("should never be used")
    node = make_generator_node(_gateway(completion_fn))
    state = initial_state("q")
    state["reranked_chunks"] = []

    result = await node(state)

    assert result["abstained"] is True
    assert calls == []


@pytest.mark.asyncio
async def test_generator_node_passes_through_model_abstention_on_irrelevant_chunks() -> None:
    chunks = [_scored("1", 0.9)]
    payload = json.dumps(
        {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "abstained": True,
            "abstain_reason": "the chunk does not answer the question",
        }
    )
    completion_fn, calls = _make_completion_fn(payload)
    node = make_generator_node(_gateway(completion_fn))
    state = initial_state("q")
    state["reranked_chunks"] = chunks

    result = await node(state)

    assert result["abstained"] is True
    assert result["abstain_reason"] == "the chunk does not answer the question"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_generator_node_raises_on_malformed_response() -> None:
    # missing required fields (answer, confidence, abstained) -> fails GeneratedAnswer
    # validation on every attempt/provider -> Gateway exhausts the chain
    completion_fn, _ = _make_completion_fn('{"citations": []}')
    node = make_generator_node(_gateway(completion_fn, same_provider_retry_attempts=1))
    state = initial_state("q")
    state["reranked_chunks"] = [_scored("1", 0.9)]

    with pytest.raises(AllProvidersExhausted):
        await node(state)


# --- critic node ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_node_passes_faithful_answer() -> None:
    payload = json.dumps({"faithful": True})
    completion_fn, calls = _make_completion_fn(payload)
    node = make_critic_node(_gateway(completion_fn))
    state = initial_state("q")
    state["reranked_chunks"] = [_scored("1", 0.9)]
    state["answer"] = "The answer is X, per chunk 1."
    state["citations"] = ["1"]
    state["abstained"] = False

    result = await node(state)

    assert result["needs_retry"] is False
    assert result["critic_feedback"] is None
    assert "retry_count" not in result  # not incremented on pass
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_critic_node_fails_fabricated_claim_and_writes_reason() -> None:
    payload = json.dumps(
        {"faithful": False, "reason": "answer includes a claim not supported by chunk 1"}
    )
    completion_fn, _ = _make_completion_fn(payload)
    node = make_critic_node(_gateway(completion_fn))
    state = initial_state("q")
    state["reranked_chunks"] = [_scored("1", 0.9)]
    state["answer"] = "The answer claims something chunk 1 doesn't say."
    state["citations"] = ["1"]
    state["abstained"] = False
    state["retry_count"] = 0

    result = await node(state)

    assert result["needs_retry"] is True
    assert result["retry_count"] == 1
    assert result["critic_feedback"] == "answer includes a claim not supported by chunk 1"


@pytest.mark.asyncio
async def test_critic_node_skips_gateway_call_on_abstained_answer() -> None:
    completion_fn, calls = _make_completion_fn("should never be used")
    node = make_critic_node(_gateway(completion_fn))
    state = initial_state("q")
    state["abstained"] = True

    result = await node(state)

    assert result["needs_retry"] is False
    assert calls == []


@pytest.mark.asyncio
async def test_critic_node_output_respects_hard_cap_via_route_after_critic() -> None:
    payload = json.dumps({"faithful": False, "reason": "still not faithful"})
    completion_fn, _ = _make_completion_fn(payload)
    node = make_critic_node(_gateway(completion_fn))
    state = initial_state("q", max_retries=2)
    state["reranked_chunks"] = [_scored("1", 0.9)]
    state["answer"] = "some claim"
    state["citations"] = ["1"]

    for _ in range(state["max_retries"]):
        update = await node(state)
        state.update(update)  # type: ignore[typeddict-item]
        expected = "retry" if state["retry_count"] < state["max_retries"] else "proceed"
        assert route_after_critic(state) == expected

    assert route_after_critic(state) == "proceed"
