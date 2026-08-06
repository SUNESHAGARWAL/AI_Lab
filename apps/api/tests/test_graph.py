import pytest
from api.graph.build import build_graph
from api.graph.nodes import route_after_critic
from api.graph.state import initial_state
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


def _graph():
    return build_graph(InMemorySaver())


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
