"""SSE endpoint that runs a query through the graph and streams typed
api.graph.events.GraphEvent objects as execution progresses — the streaming contract
the frontend's live graph visualization consumes. The non-streaming graph
(app.state.graph, built once in api.main.lifespan) is untouched; this route builds its
own CompiledStateGraph per request so each request gets an isolated
api.graph.streaming.RecordingGateway (see that module for why isolation matters)."""

import time
from collections.abc import AsyncIterator
from typing import cast

from core.models import Filters
from fastapi import APIRouter, Request
from opentelemetry import trace
from pydantic import BaseModel, Field
from starlette.datastructures import State
from starlette.responses import StreamingResponse

from api.config import get_settings
from api.graph.build import build_graph
from api.graph.events import GraphErrorEvent, GraphStartedEvent
from api.graph.state import initial_state
from api.graph.streaming import RecordingGateway, new_thread_id, stream_graph_events
from llm import Gateway

router = APIRouter()
_tracer = trace.get_tracer(__name__)
_settings = get_settings()  # module-level, same eager-fail-loudly pattern as api.main


class StreamQueryRequest(BaseModel):
    # The demo endpoint is public — a length cap is the input guard required by
    # CLAUDE.md's security rules for every public path. Config-driven (max_query_length
    # in api.config.Settings), not a bare constant, so it's tunable without a redeploy.
    query: str = Field(min_length=1, max_length=_settings.max_query_length)
    thread_id: str | None = None
    filters: Filters | None = None
    max_retries: int | None = None


def _format_sse(seq: int, event: BaseModel) -> str:
    event_type = event.model_dump()["type"]
    return f"id: {seq}\nevent: {event_type}\ndata: {event.model_dump_json()}\n\n"


async def _sse_body(
    request: StreamQueryRequest, app_state: State, client_key: str
) -> AsyncIterator[str]:
    thread_id = request.thread_id or new_thread_id()

    # Rate-limit before doing any work at all — no graph build, no gateway call, not
    # even a thread checkpoint. Cached example replays never hit this route (see
    # apps/web/lib/replay-client.ts), so only genuinely live queries are counted.
    limit_result = await app_state.rate_limiter.check(client_key)
    if not limit_result.allowed:
        yield _format_sse(
            0, GraphStartedEvent(thread_id=thread_id, emitted_at=time.time(), query=request.query)
        )
        yield _format_sse(
            1,
            GraphErrorEvent(
                thread_id=thread_id,
                emitted_at=time.time(),
                message=(
                    "You've hit the free live-query limit for now — try one of the "
                    "example questions below, they're always free."
                ),
                retryable=True,
                reason="rate_limited",
            ),
        )
        return

    recorder = RecordingGateway(app_state.gateway)
    graph = build_graph(
        app_state.checkpointer,
        app_state.retriever,
        app_state.reranker,
        cast(Gateway, recorder),
    )
    kwargs = {} if request.max_retries is None else {"max_retries": request.max_retries}
    state = initial_state(request.query, filters=request.filters, **kwargs)
    config = {"configurable": {"thread_id": thread_id}}

    with _tracer.start_as_current_span("api.stream_query") as span:
        span.set_attribute("thread_id", thread_id)
        seq = 0
        async for event in stream_graph_events(graph, state, config, recorder):
            yield _format_sse(seq, event)
            seq += 1


@router.post("/query/stream")
async def stream_query(payload: StreamQueryRequest, request: Request) -> StreamingResponse:
    # No trusted-proxy/X-Forwarded-For handling yet — request.client.host is the
    # connecting socket's address, correct for direct local/dev traffic but will read
    # as the proxy's IP once this sits behind one. Deploy-time follow-up, not this pass.
    client_key = request.client.host if request.client else "unknown"
    return StreamingResponse(
        _sse_body(payload, request.app.state, client_key), media_type="text/event-stream"
    )
