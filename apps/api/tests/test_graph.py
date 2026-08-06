import json
from types import SimpleNamespace
from typing import Any

import pytest
from api.graph.build import build_graph
from api.graph.nodes import route_after_critic
from api.graph.schemas import CriticVerdict
from api.graph.state import initial_state
from core.models import Chunk
from core.testing import FakeReranker, FakeRetriever
from fakeredis import FakeAsyncRedis
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from llm import Gateway


def _fake_gateway() -> Gateway:
    generated_payload = json.dumps(
        {"answer": "stub answer", "citations": [], "confidence": 0.5, "abstained": False}
    )
    # Always fails, reproducing the old stub critic's "always wants another pass"
    # behavior through the real critic node — proves the hard cap still holds.
    critic_payload = json.dumps({"faithful": False, "reason": "stub: always retry"})

    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        # Gateway._call_provider forwards response_model as response_format, so the
        # same fake can serve both the generator's and critic's distinct schemas.
        content = (
            critic_payload if kwargs.get("response_format") is CriticVerdict else generated_payload
        )
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
