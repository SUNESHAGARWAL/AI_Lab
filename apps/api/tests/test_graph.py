import json
from types import SimpleNamespace
from typing import Any

import pytest
from api.graph.build import build_graph
from api.graph.nodes import route_after_critic
from api.graph.state import initial_state
from core.models import Chunk, Query, ScoredChunk
from core.ports import Reranker, Retriever
from core.testing import FakeReranker, FakeRetriever
from fakeredis import FakeAsyncRedis
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from llm import Gateway


def _system_content(messages: list[dict[str, Any]]) -> str:
    return next((m["content"] for m in messages if m.get("role") == "system"), "")


def _node_for_messages(messages: list[dict[str, Any]]) -> str:
    """Every node (planner/generator/critic) now goes through llm.complete_json —
    none of them set response_format anymore (see make_planner_node/
    make_generator_node/make_critic_node's docstrings: neither Groq's small models
    nor DeepSeek's current API reliably serve a response_format request). So test
    fakes distinguish which node is calling by its real, distinctive system prompt
    text instead of by response_format's type."""
    system = _system_content(messages)
    if "query planner" in system:
        return "planner"
    if "fact-checker" in system:
        return "critic"
    return "generator"


def _fake_gateway() -> Gateway:
    # rewritten_query must still share terms with the fixture corpus below — see
    # _graph()'s comment.
    planner_payload = json.dumps(
        {"rewritten_query": "what is x?", "intent": "factual_lookup", "retry_budget": 2}
    )
    generated_payload = json.dumps(
        {"answer": "stub answer", "citations": [], "confidence": 0.5, "abstained": False}
    )
    # Always fails, reproducing the old stub critic's "always wants another pass"
    # behavior through the real critic node — proves the hard cap still holds.
    critic_payload = json.dumps({"faithful": False, "reason": "stub: always retry"})

    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        node = _node_for_messages(kwargs["messages"])
        content = {
            "planner": planner_payload,
            "generator": generated_payload,
            "critic": critic_payload,
        }[node]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    return Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)


def _graph():
    # Text must actually share terms with the "what is x?" query used below —
    # FakeRetriever's term-match scoring returns nothing for a non-matching corpus,
    # which would make the real generator node abstain deterministically (correct
    # behavior) and short-circuit the critic before the retry loop is ever exercised.
    corpus = [Chunk(id="1", document_id="doc-a", text="x is a placeholder chunk")]
    return build_graph(
        InMemorySaver(), FakeRetriever(corpus=corpus), FakeReranker(), _fake_gateway()
    )


def test_route_after_critic_retries_under_cap() -> None:
    state = initial_state("q", max_retries=2)
    state["needs_retry"] = True
    state["retry_count"] = 1
    assert route_after_critic(state) == "retry"


def test_route_after_critic_proceeds_at_cap() -> None:
    state = initial_state("q", max_retries=2)
    state["needs_retry"] = True
    state["retry_count"] = 2
    assert route_after_critic(state) == "proceed"


def test_route_after_critic_proceeds_when_satisfied() -> None:
    state = initial_state("q", max_retries=2)
    state["needs_retry"] = False
    state["retry_count"] = 0
    assert route_after_critic(state) == "proceed"


@pytest.mark.asyncio
async def test_graph_loops_up_to_cap_then_pauses_at_hitl_gate() -> None:
    graph = _graph()
    config = {"configurable": {"thread_id": "test-1"}}

    result = await graph.ainvoke(initial_state("what is x?", max_retries=2), config=config)

    assert result["retry_count"] == 2  # hard cap stopped the stub critic's retry loop
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("hitl_gate",)


@pytest.mark.asyncio
async def test_graph_resumes_after_hitl_gate_interrupt() -> None:
    graph = _graph()
    config = {"configurable": {"thread_id": "test-2"}}
    await graph.ainvoke(initial_state("what is x?", max_retries=1), config=config)

    result = await graph.ainvoke(Command(resume=True), config=config)

    assert result["human_approved"] is True
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()  # graph fully completed


class _SpyRetriever(Retriever):
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: Query) -> list[ScoredChunk]:
        self.calls += 1
        return []


class _SpyReranker(Reranker):
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: Query, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        self.calls += 1
        return candidates


@pytest.mark.asyncio
async def test_graph_short_circuits_on_out_of_scope_intent() -> None:
    planner_payload = json.dumps(
        {
            "rewritten_query": "off topic",
            "intent": "out_of_scope",
            "retry_budget": 0,
            "abstain_reason": "not answerable by this system",
        }
    )
    calls_by_node: list[str] = []

    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        calls_by_node.append(_node_for_messages(kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=planner_payload))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    gateway = Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)
    retriever = _SpyRetriever()
    reranker = _SpyReranker()
    graph = build_graph(InMemorySaver(), retriever, reranker, gateway)
    config = {"configurable": {"thread_id": "oos-1"}}

    result = await graph.ainvoke(initial_state("asdkjaslkdj nonsense"), config=config)

    assert result["abstained"] is True
    assert retriever.calls == 0
    assert reranker.calls == 0
    assert calls_by_node == ["planner"]  # generator/critic never invoked
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("hitl_gate",)

    # out-of-scope framing: no draft to review, distinct from the low-confidence case
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["type"] == "out_of_scope"
    assert "citations" not in interrupt_value


def _review_gateway() -> Gateway:
    """A non-abstained, low-confidence, cited answer that the critic accepts
    immediately (faithful=True, no retry loop) — reaches hitl_gate via the normal
    critic 'proceed' path with the draft still intact, to exercise the review
    framing (as opposed to the out-of-scope framing above)."""
    planner_payload = json.dumps(
        {"rewritten_query": "what is x?", "intent": "factual_lookup", "retry_budget": 1}
    )
    generated_payload = json.dumps(
        {
            "answer": "draft answer citing chunk 1",
            "citations": ["1"],
            "confidence": 0.3,
            "abstained": False,
        }
    )
    critic_payload = json.dumps({"faithful": True})

    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        node = _node_for_messages(kwargs["messages"])
        content = {
            "planner": planner_payload,
            "generator": generated_payload,
            "critic": critic_payload,
        }[node]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    return Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)


def _review_graph():
    corpus = [Chunk(id="1", document_id="doc-a", text="x is a placeholder chunk")]
    return build_graph(
        InMemorySaver(), FakeRetriever(corpus=corpus), FakeReranker(), _review_gateway()
    )


@pytest.mark.asyncio
async def test_hitl_gate_triggers_review_framing_on_low_confidence() -> None:
    graph = _review_graph()
    config = {"configurable": {"thread_id": "review-1"}}

    result = await graph.ainvoke(initial_state("what is x?"), config=config)

    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["type"] == "review"
    assert interrupt_value["answer"] == "draft answer citing chunk 1"
    assert interrupt_value["citations"] == ["1"]
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("hitl_gate",)


@pytest.mark.asyncio
async def test_hitl_gate_resume_accept_finalizes_state() -> None:
    graph = _review_graph()
    config = {"configurable": {"thread_id": "review-2"}}
    await graph.ainvoke(initial_state("what is x?"), config=config)

    result = await graph.ainvoke(Command(resume={"decision": "accept"}), config=config)

    assert result["human_approved"] is True
    assert result["answer"] == "draft answer citing chunk 1"
    assert result["citations"] == ["1"]


@pytest.mark.asyncio
async def test_hitl_gate_resume_reject_finalizes_state() -> None:
    graph = _review_graph()
    config = {"configurable": {"thread_id": "review-3"}}
    await graph.ainvoke(initial_state("what is x?"), config=config)

    result = await graph.ainvoke(Command(resume={"decision": "reject"}), config=config)

    assert result["human_approved"] is False


@pytest.mark.asyncio
async def test_hitl_gate_resume_edit_overwrites_answer_and_citations() -> None:
    graph = _review_graph()
    config = {"configurable": {"thread_id": "review-4"}}
    await graph.ainvoke(initial_state("what is x?"), config=config)

    result = await graph.ainvoke(
        Command(resume={"decision": "edit", "answer": "corrected answer", "citations": ["9"]}),
        config=config,
    )

    assert result["human_approved"] is True
    assert result["answer"] == "corrected answer"
    assert result["citations"] == ["9"]
