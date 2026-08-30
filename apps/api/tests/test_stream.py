import json
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest
from api.config import Settings
from api.graph.build import build_graph
from api.graph.events import GraphEvent, NodeName
from api.graph.state import initial_state
from api.graph.streaming import RecordingGateway, stream_graph_events
from api.ratelimit import RateLimiter, RateLimitResult
from api.routes.stream import StreamQueryRequest, _guarded_sse_body, _sse_body
from core.models import Chunk
from core.testing import FakeReranker, FakeRetriever
from fakeredis import FakeAsyncRedis
from langgraph.checkpoint.memory import InMemorySaver
from llm.config import GatewaySettings

from llm import AllProvidersExhausted, Gateway, Tier


def _node_for_messages(messages: list[dict[str, Any]]) -> str:
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    if "query planner" in system:
        return "planner"
    if "fact-checker" in system:
        return "critic"
    return "generator"


def _completion_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _review_gateway() -> Gateway:
    """Non-abstained, low-confidence answer the critic accepts immediately — reaches
    hitl_gate via the normal 'proceed' path with the draft intact, same fixture shape
    as test_graph.py's _review_gateway()."""
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
        return _completion_response(content)

    return Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)


def _always_retry_gateway() -> Gateway:
    """Critic always fails, reproducing the hard-cap retry-loop scenario already
    covered non-streaming in test_graph.py's test_graph_loops_up_to_cap_then_pauses_at_hitl_gate."""
    planner_payload = json.dumps(
        {"rewritten_query": "what is x?", "intent": "factual_lookup", "retry_budget": 2}
    )
    generated_payload = json.dumps(
        {"answer": "stub answer", "citations": [], "confidence": 0.5, "abstained": False}
    )
    critic_payload = json.dumps({"faithful": False, "reason": "stub: always retry"})

    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        node = _node_for_messages(kwargs["messages"])
        content = {
            "planner": planner_payload,
            "generator": generated_payload,
            "critic": critic_payload,
        }[node]
        return _completion_response(content)

    return Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)


def _out_of_scope_gateway() -> Gateway:
    planner_payload = json.dumps(
        {
            "rewritten_query": "off topic",
            "intent": "out_of_scope",
            "retry_budget": 0,
            "abstain_reason": "not answerable by this system",
        }
    )

    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        return _completion_response(planner_payload)

    return Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)


def _exhausted_gateway() -> Gateway:
    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        raise AllProvidersExhausted(Tier.FAST)

    return Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)


def _budget_exhausted_gateway() -> Gateway:
    """A per_day_token_ceiling so tight the planner's first call blows straight through
    it — reproduces the demo's hard daily cost ceiling being hit mid-stream, same shape
    as _exhausted_gateway() above but scoped "per_day" instead of AllProvidersExhausted."""

    async def completion_fn(**kwargs: Any) -> SimpleNamespace:
        return _completion_response("unreachable — budget guard blocks before this runs")

    return Gateway(
        settings=GatewaySettings(per_day_token_ceiling=1),
        redis_client=FakeAsyncRedis(),
        completion_fn=completion_fn,
    )


def _corpus() -> list[Chunk]:
    return [Chunk(id="1", document_id="doc-a", text="x is a placeholder chunk")]


async def _run(
    gateway: Gateway, thread_id: str, *, max_retries: int = 2, query: str = "what is x?"
) -> list[GraphEvent]:
    recorder = RecordingGateway(gateway)
    graph = build_graph(InMemorySaver(), FakeRetriever(corpus=_corpus()), FakeReranker(), recorder)
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(query, max_retries=max_retries)
    return [event async for event in stream_graph_events(graph, state, config, recorder)]


@pytest.mark.asyncio
async def test_normal_query_emits_full_node_sequence_with_token_usage() -> None:
    events = await _run(_review_gateway(), "stream-1", max_retries=1)

    assert events[0].type == "graph_started"
    node_events = [e for e in events if e.type in ("node_started", "node_completed")]
    expected_order = [
        NodeName.PLANNER,
        NodeName.RETRIEVER,
        NodeName.RERANKER,
        NodeName.GENERATOR,
        NodeName.CRITIC,
        NodeName.HITL_GATE,
    ]
    started_order = [e.node for e in node_events if e.type == "node_started"]
    assert started_order == expected_order

    completed_by_node = {e.node: e for e in node_events if e.type == "node_completed"}
    for node in (NodeName.PLANNER, NodeName.GENERATOR, NodeName.CRITIC):
        assert len(completed_by_node[node].llm_calls) >= 1
        assert completed_by_node[node].latency_ms >= 0
    for node in (NodeName.RETRIEVER, NodeName.RERANKER, NodeName.HITL_GATE):
        assert completed_by_node[node].llm_calls == []

    # confidence 0.3 < DEFAULT_CONFIDENCE_THRESHOLD -> pauses for human review
    assert events[-1].type == "graph_interrupted"
    assert events[-1].interrupt["type"] == "review"
    assert events[-1].interrupt["answer"] == "draft answer citing chunk 1"
    assert not any(e.type == "retry_loop" for e in events)


@pytest.mark.asyncio
async def test_retry_loop_emitted_and_capped() -> None:
    events = await _run(_always_retry_gateway(), "stream-2", max_retries=2)

    retry_events = [e for e in events if e.type == "retry_loop"]
    assert [e.retry_count for e in retry_events] == [1, 2]

    run_ids_by_node: dict[NodeName, set[str]] = {}
    for e in events:
        if e.type == "node_started":
            run_ids_by_node.setdefault(e.node, set()).add(e.run_id)
    for node in (NodeName.RETRIEVER, NodeName.RERANKER, NodeName.GENERATOR, NodeName.CRITIC):
        # retry_count is incremented before route_after_critic's cap check (see
        # api.graph.nodes.route_after_critic), so cap=2 permits exactly 2 total
        # passes (retry_count reaches 1 -> loop, reaches 2 -> cap hit, proceed).
        assert len(run_ids_by_node[node]) == 2

    assert events[-1].type == "graph_interrupted"  # cap hit -> low confidence -> review gate


@pytest.mark.asyncio
async def test_out_of_scope_short_circuits_before_retrieval() -> None:
    events = await _run(_out_of_scope_gateway(), "stream-3")

    started_nodes = [e.node for e in events if e.type == "node_started"]
    assert started_nodes == [NodeName.PLANNER, NodeName.HITL_GATE]

    completed = [e for e in events if e.type == "node_completed"]
    assert all(e.llm_calls == [] for e in completed if e.node is NodeName.HITL_GATE)

    assert events[-1].type == "graph_interrupted"
    assert events[-1].interrupt["type"] == "out_of_scope"
    assert "citations" not in events[-1].interrupt


@pytest.mark.asyncio
async def test_gateway_exhaustion_emits_error_event_and_terminates_cleanly() -> None:
    events = await _run(_exhausted_gateway(), "stream-4")

    assert events[-1].type == "error"
    assert events[-1].retryable is True
    assert events[-1].reason == "provider_exhausted"
    assert not any(e.type in ("graph_completed", "graph_interrupted") for e in events)


@pytest.mark.asyncio
async def test_daily_budget_exhaustion_degrades_gracefully() -> None:
    """The demo's hard cost ceiling (packages/llm's BudgetGuard, per_day scope) must
    fail soft — a friendly, structured error event the frontend can point a visitor at
    the free examples with, never a raw exception or a 500."""
    events = await _run(_budget_exhausted_gateway(), "stream-5")

    assert events[-1].type == "error"
    assert events[-1].retryable is False
    assert events[-1].reason == "budget_exhausted"
    assert "example question" in events[-1].message
    assert not any(e.type in ("graph_completed", "graph_interrupted") for e in events)


class _DenyingRateLimiter:
    async def check(self, client_key: str) -> RateLimitResult:
        return RateLimitResult(allowed=False, remaining=0, limit=5)


@pytest.mark.asyncio
async def test_rate_limited_request_never_builds_the_graph() -> None:
    """app_state deliberately has no checkpointer/retriever/reranker/gateway — if
    _sse_body tried to build_graph() anyway this would blow up with an AttributeError
    before yielding anything, instead of the two well-formed SSE frames asserted here."""
    app_state = SimpleNamespace(rate_limiter=_DenyingRateLimiter())
    request = StreamQueryRequest(query="what is x?")

    frames = [frame async for frame in _sse_body(request, app_state, "9.9.9.9")]

    assert len(frames) == 2
    assert "event: graph_started" in frames[0]
    assert "event: error" in frames[1]
    assert '"reason":"rate_limited"' in frames[1]
    assert '"retryable":true' in frames[1]


class _ExplodingRateLimiter:
    async def check(self, client_key: str) -> RateLimitResult:
        raise AssertionError("a greeting must not consume a live-query allowance")


@pytest.mark.asyncio
async def test_greeting_short_circuits_before_the_rate_limiter_and_the_graph() -> None:
    """The scope guard's cost guarantee, asserted structurally: app_state has no
    checkpointer/retriever/reranker/gateway, so any attempt to build or run the graph
    raises AttributeError, and the rate limiter raises if consulted. Two clean frames
    means zero gateway calls and zero allowance spent."""
    app_state = SimpleNamespace(rate_limiter=_ExplodingRateLimiter())
    request = StreamQueryRequest(query="hi")

    frames = [frame async for frame in _sse_body(request, app_state, "9.9.9.9")]

    assert len(frames) == 2
    assert "event: graph_started" in frames[0]
    assert "event: graph_interrupted" in frames[1]
    assert '"type":"out_of_scope"' in frames[1]


@pytest.mark.asyncio
async def test_real_question_is_not_short_circuited_by_the_scope_guard() -> None:
    """The complement of the test above: the same bare app_state must now fail trying to
    reach the rate limiter, proving a genuine compliance question still takes the full
    path rather than being answered by the guard."""
    app_state = SimpleNamespace(rate_limiter=_ExplodingRateLimiter())
    request = StreamQueryRequest(
        query="what does the EU AI Act mention on personal information handling"
    )

    with pytest.raises(AssertionError, match="live-query allowance"):
        [frame async for frame in _sse_body(request, app_state, "9.9.9.9")]


class _DeadConnectionSaver(InMemorySaver):
    """A checkpointer whose very first write fails the way a severed Postgres
    connection does — the exact production fault this guards against: Neon suspended
    its free-tier compute, the checkpointer's single long-lived connection died with
    it, and `aput` raised before the planner was ever scheduled."""

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        raise psycopg.OperationalError("consuming input failed: server closed the connection")


@pytest.mark.asyncio
async def test_unexpected_infrastructure_failure_still_emits_a_terminal_event() -> None:
    """The stream must never just stop. Before this, any exception other than
    AllProvidersExhausted/BudgetExceeded escaped the async generator and
    StreamingResponse simply closed: the client got a 200 that ended after
    `graph_started` with no terminal event, and the frontend could only guess. A
    dropped checkpointer connection therefore looked like a frontend bug for as long
    as it took someone to redeploy."""
    recorder = RecordingGateway(_review_gateway())
    graph = build_graph(
        _DeadConnectionSaver(), FakeRetriever(corpus=_corpus()), FakeReranker(), recorder
    )
    config = {"configurable": {"thread_id": "stream-dead-conn"}}
    state = initial_state("what is x?", max_retries=1)

    events = [event async for event in stream_graph_events(graph, state, config, recorder)]

    assert events[0].type == "graph_started"
    assert events[-1].type == "error"
    assert events[-1].reason == "internal_error"
    assert events[-1].retryable is True
    # The visitor-facing message must not leak the driver's exception text.
    assert "server closed the connection" not in events[-1].message
    assert not any(e.type in ("graph_completed", "graph_interrupted") for e in events)


class _BrokenRateLimiter:
    """Redis unreachable — what a misconfigured or down Upstash actually looks like
    from inside the rate limiter."""

    async def check(self, client_key: str) -> RateLimitResult:
        raise ConnectionError("Connection closed by server.")


@pytest.mark.asyncio
async def test_failure_outside_the_graph_still_emits_a_terminal_frame() -> None:
    """The companion to the test above, for the other side of the boundary. A fault in
    the rate limiter happens before stream_graph_events is ever entered, so its internal
    handler cannot see it; unguarded, the request completed with a 200 and a completely
    empty body. _guarded_sse_body is what turns that into a terminal error frame."""
    app_state = SimpleNamespace(rate_limiter=_BrokenRateLimiter())
    request = StreamQueryRequest(query="what is x?")

    frames = [frame async for frame in _guarded_sse_body(request, app_state, "9.9.9.9")]

    assert len(frames) == 1
    assert "event: error" in frames[0]
    assert '"reason":"internal_error"' in frames[0]
    assert '"retryable":true' in frames[0]
    # The driver's exception text must not reach the visitor.
    assert "Connection closed by server" not in frames[0]


def _refund_app_state(gateway: Gateway, redis: FakeAsyncRedis, limit: int = 5) -> SimpleNamespace:
    settings = Settings(
        database_url="postgresql://x", redis_url="redis://x", live_query_rate_limit_per_hour=limit
    )
    return SimpleNamespace(
        rate_limiter=RateLimiter(redis, settings),
        checkpointer=InMemorySaver(),
        retriever=FakeRetriever(corpus=_corpus()),
        reranker=FakeReranker(),
        gateway=gateway,
    )


async def _spend_one(app_state: SimpleNamespace, gateway_query: str = "what is x?") -> None:
    request = StreamQueryRequest(query=gateway_query, max_retries=1)
    async for _frame in _guarded_sse_body(request, app_state, "5.5.5.5"):
        pass


async def _consumed(redis: FakeAsyncRedis) -> int:
    keys = [k async for k in redis.scan_iter("api:ratelimit:*")]
    return int(await redis.get(keys[0])) if keys else 0


@pytest.mark.asyncio
async def test_our_own_failure_does_not_cost_the_visitor_an_allowance() -> None:
    """The outage made this concrete: with the backend broken, a visitor could burn
    every live query they had on requests that never returned an answer. A run that
    fails through no fault of theirs has to give the allowance back."""
    redis = FakeAsyncRedis()
    app_state = _refund_app_state(_exhausted_gateway(), redis)

    await _spend_one(app_state)

    assert await _consumed(redis) == 0


@pytest.mark.asyncio
async def test_a_delivered_answer_still_costs_an_allowance() -> None:
    """The other half of the policy — refunding real work would make the limit
    meaningless. An abstention counts as real work too: it is the product."""
    redis = FakeAsyncRedis()
    app_state = _refund_app_state(_review_gateway(), redis)

    await _spend_one(app_state)

    assert await _consumed(redis) == 1


@pytest.mark.asyncio
async def test_the_daily_cost_ceiling_is_never_refunded() -> None:
    """budget_exhausted is the one limit that actually caps spend. Refunding it would
    make it infinitely retryable, which defeats the entire point of having it."""
    redis = FakeAsyncRedis()
    app_state = _refund_app_state(_budget_exhausted_gateway(), redis)

    await _spend_one(app_state)

    assert await _consumed(redis) == 1


@pytest.mark.asyncio
async def test_a_visitor_disconnecting_mid_stream_does_not_break_the_refund_path() -> None:
    """Abandoning the stream raises GeneratorExit at the yield. Awaiting a refund while
    unwinding that would raise "async generator ignored GeneratorExit" — noise on a
    path that is already just cleanup."""
    redis = FakeAsyncRedis()
    app_state = _refund_app_state(_review_gateway(), redis)
    request = StreamQueryRequest(query="what is x?", max_retries=1)

    body = _guarded_sse_body(request, app_state, "5.5.5.5")
    await body.__anext__()  # take the first frame, then walk away
    await body.aclose()  # must not raise

    # Work had already started, so the allowance stays spent.
    assert await _consumed(redis) == 1
