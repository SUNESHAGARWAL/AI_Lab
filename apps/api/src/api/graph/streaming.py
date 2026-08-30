"""Turns LangGraph's native task-level streaming into the typed events in
api.graph.events. Verified against langgraph==1.2.10 (see docs/adr — this module's
design note in the implementation plan): graph.astream(state, config,
stream_mode=["tasks"]) yields a start payload ({"id","name","input","triggers"}) when a
node is scheduled and a result payload ({"id","name","error","result","interrupts"})
when it finishes, where "result" is exactly the dict the node function returned and
"interrupts" is populated the moment interrupt() fires inside hitl_gate — no need to
separately poll aget_state() for the interrupt payload, and no changes to any node
function were needed to get here.

Token cost/usage isn't part of that stream (nodes call llm.complete_json, which
discards CompletionResult after parsing) — RecordingGateway captures it by wrapping the
real Gateway per request; see the module docstring on that class for why this is a
request-scoped wrapper rather than a change to packages/llm/gateway.py."""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from api.graph.events import (
    NODE_PAYLOAD_BUILDERS,
    GraphCompletedEvent,
    GraphErrorEvent,
    GraphEvent,
    GraphInterruptedEvent,
    GraphStartedEvent,
    NodeCompletedEvent,
    NodeName,
    NodeStartedEvent,
    RetryLoopEvent,
    TokenUsage,
)
from api.graph.state import AgentState
from llm import AllProvidersExhausted, BudgetExceeded, CompletionRequest, CompletionResult, Gateway
from telemetry import get_logger

logger = get_logger(__name__)


class RecordingGateway:
    """Wraps a real llm.Gateway to capture every CompletionResult it returns, scoped to
    one SSE request. Duck-types the one method llm.complete_json actually calls
    (`complete`) — passed to build_graph via `cast(Gateway, ...)`, the same shape as
    this codebase's tests already inject a completion_fn-backed fake Gateway one layer
    down. Safe under concurrent SSE requests because each request gets its own instance
    and its own compiled graph (see api.routes.stream) — no shared mutable state."""

    def __init__(self, inner: Gateway) -> None:
        self._inner = inner
        self._calls: list[CompletionResult] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        result = await self._inner.complete(request)
        self._calls.append(result)
        return result

    def drain(self) -> list[CompletionResult]:
        calls, self._calls = self._calls, []
        return calls


def _to_token_usage(result: CompletionResult) -> TokenUsage:
    return TokenUsage(
        provider=result.provider,
        model=result.model,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        latency_ms=result.latency_ms,
    )


async def stream_graph_events(
    graph: CompiledStateGraph,
    state: AgentState,
    config: RunnableConfig,
    recorder: RecordingGateway,
) -> AsyncIterator[GraphEvent]:
    """Drives one graph run and yields typed events in real time. `config` must carry
    a "thread_id" (same contract as graph.ainvoke — see api.graph.checkpointer)."""
    thread_id = str(config["configurable"]["thread_id"])
    yield GraphStartedEvent(thread_id=thread_id, emitted_at=time.time(), query=state["query"])

    started_at: dict[str, float] = {}
    interrupted = False

    try:
        async for _mode, payload in graph.astream(state, config, stream_mode=["tasks"]):
            task_id = str(payload["id"])
            node = NodeName(payload["name"])

            if "result" not in payload:
                started_at[task_id] = time.time()
                yield NodeStartedEvent(
                    thread_id=thread_id, emitted_at=time.time(), node=node, run_id=task_id
                )
                continue

            start = started_at.pop(task_id, time.time())
            latency_ms = (time.time() - start) * 1000
            result: dict[str, Any] = payload.get("result") or {}
            llm_calls = [_to_token_usage(r) for r in recorder.drain()]
            build_payload = NODE_PAYLOAD_BUILDERS[node]

            yield NodeCompletedEvent(
                thread_id=thread_id,
                emitted_at=time.time(),
                node=node,
                run_id=task_id,
                latency_ms=latency_ms,
                llm_calls=llm_calls,
                payload=build_payload(result),
            )

            if node is NodeName.CRITIC and result.get("needs_retry"):
                yield RetryLoopEvent(
                    thread_id=thread_id,
                    emitted_at=time.time(),
                    retry_count=result["retry_count"],
                    reason=result.get("critic_feedback"),
                )

            interrupts = payload.get("interrupts") or []
            if interrupts:
                interrupted = True
                yield GraphInterruptedEvent(
                    thread_id=thread_id,
                    emitted_at=time.time(),
                    interrupt=interrupts[0]["value"],
                )
    except (AllProvidersExhausted, BudgetExceeded) as exc:
        # A per-day ceiling is the demo's hard cost cap and gets a visitor-facing
        # message pointing at the always-free examples; a per-request ceiling or an
        # exhausted provider chain is a real fault, not a "you asked too much"
        # message, so it keeps the raw exception text.
        if isinstance(exc, BudgetExceeded) and exc.scope == "per_day":
            message = (
                "Live demo budget reached for today — try one of the example "
                "questions, they're always free."
            )
            reason = "budget_exhausted"
        elif isinstance(exc, AllProvidersExhausted):
            # Named explicitly rather than left as None. Every provider in the chain
            # refusing is our infrastructure failing, not the visitor's query being
            # unreasonable, and the route has to tell the two apart to decide whether
            # to refund the live-query allowance (api.routes.stream). Leaving both as
            # None made a discriminator that did not discriminate.
            message = str(exc)
            reason = "provider_exhausted"
        else:
            # A per-request budget ceiling: this particular query was too expensive.
            message = str(exc)
            reason = None
        yield GraphErrorEvent(
            thread_id=thread_id,
            emitted_at=time.time(),
            message=message,
            retryable=isinstance(exc, AllProvidersExhausted),
            reason=reason,
        )
        return
    except Exception:
        # Anything else — a dropped Postgres connection on the checkpoint write, a
        # retriever fault, a bug in a node. Without this the exception escapes the
        # async generator and StreamingResponse simply stops sending: the client sees
        # a 200 that ends after `graph_started`, with no terminal event and nothing in
        # the payload to explain it. That is a silent outage, and it is exactly how the
        # dead-checkpointer-connection bug hid (see api.graph.checkpointer). Every exit
        # from this generator now carries a terminal event.
        #
        # exc_info, not the message text: the exception can carry a connection string
        # or prompt fragment, and CLAUDE.md keeps those out of the response body. The
        # visitor gets a generic retryable error; the detail goes to the logs.
        logger.exception("stream.graph_failed", thread_id=thread_id)
        yield GraphErrorEvent(
            thread_id=thread_id,
            emitted_at=time.time(),
            message=(
                "Something broke on our side partway through answering. "
                "Try again, or use one of the example questions — those always work."
            ),
            retryable=True,
            reason="internal_error",
        )
        return

    if not interrupted:
        snapshot = await graph.aget_state(config)
        values = snapshot.values
        yield GraphCompletedEvent(
            thread_id=thread_id,
            emitted_at=time.time(),
            answer=values.get("answer"),
            citations=values.get("citations", []),
            confidence=values.get("confidence"),
            abstained=values.get("abstained", False),
            human_approved=values.get("human_approved"),
        )


def new_thread_id() -> str:
    return str(uuid.uuid4())
