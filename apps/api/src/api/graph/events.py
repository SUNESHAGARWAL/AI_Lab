"""Typed SSE event schema for streaming graph execution — see api.graph.streaming for
the translator that turns LangGraph's native astream(stream_mode=["tasks"]) output into
these events. This is the shared contract the frontend's TypeScript types will later
mirror, so every field here is something the UI actually needs to animate the graph,
not an internal implementation detail."""

from enum import StrEnum
from typing import Any, Literal

from core.models import ScoredChunk
from pydantic import BaseModel, ConfigDict, Field

from api.graph.schemas import QueryIntent


class NodeName(StrEnum):
    """Matches the node names registered in api.graph.build.build_graph exactly —
    these are LangGraph's own node names, not a separate naming scheme."""

    PLANNER = "planner"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    GENERATOR = "generator"
    CRITIC = "critic"
    HITL_GATE = "hitl_gate"


class TokenUsage(BaseModel):
    """One gateway call's telemetry, captured by api.graph.streaming.RecordingGateway.
    A node emits zero of these (retriever/reranker/hitl_gate never call the gateway),
    one (the common case), or two (llm.complete_json's one repair retry)."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float


class GraphStartedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["graph_started"] = "graph_started"
    thread_id: str
    emitted_at: float
    query: str


class NodeStartedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["node_started"] = "node_started"
    thread_id: str
    emitted_at: float
    node: NodeName
    run_id: str


class NodeCompletedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["node_completed"] = "node_completed"
    thread_id: str
    emitted_at: float
    node: NodeName
    run_id: str
    latency_ms: float
    llm_calls: list[TokenUsage] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class RetryLoopEvent(BaseModel):
    """Fired when the critic decides to loop back to the retriever — the backward edge
    the frontend needs to animate distinctly from the normal forward flow. Derived from
    the critic node's own result dict (needs_retry=True), not a separate LangGraph
    signal — see api.graph.nodes.make_critic_node."""

    model_config = ConfigDict(frozen=True)

    type: Literal["retry_loop"] = "retry_loop"
    thread_id: str
    emitted_at: float
    from_node: Literal[NodeName.CRITIC] = NodeName.CRITIC
    to_node: Literal[NodeName.RETRIEVER] = NodeName.RETRIEVER
    retry_count: int
    reason: str | None = None


class GraphInterruptedEvent(BaseModel):
    """The graph paused inside hitl_gate's interrupt() call. `interrupt` is the raw
    payload passed to interrupt() in api.graph.nodes.make_hitl_gate_node — either the
    out_of_scope framing or the review (draft answer) framing."""

    model_config = ConfigDict(frozen=True)

    type: Literal["graph_interrupted"] = "graph_interrupted"
    thread_id: str
    emitted_at: float
    interrupt: dict[str, Any]


class GraphCompletedEvent(BaseModel):
    """The graph reached END without pausing — hitl_gate auto-approved because
    confidence was already high enough (see api.graph.nodes._needs_human_review)."""

    model_config = ConfigDict(frozen=True)

    type: Literal["graph_completed"] = "graph_completed"
    thread_id: str
    emitted_at: float
    answer: str | None
    citations: list[str]
    confidence: float | None
    abstained: bool
    human_approved: bool | None


class GraphErrorEvent(BaseModel):
    """A gateway failure (AllProvidersExhausted/BudgetExceeded) propagated out of a
    node mid-stream, a pre-graph rejection (rate limit) emitted before any node runs,
    or any other fault anywhere in the request ("internal_error" — a dropped database
    connection, an unreachable Redis). SSE can't change the HTTP status after headers
    are sent, so this is the in-band equivalent of the non-streaming path's 503/429,
    and it is the *only* way a fault can be reported once streaming has begun: without
    one of these the response just stops, which is indistinguishable from success to
    everything downstream.

    `reason` is a structured discriminator for the frontend to branch on directly
    instead of pattern-matching `message` text — `None` for the original
    AllProvidersExhausted/per-request-budget cases (unchanged since before this field
    existed), set for the demo-hardening cases added alongside it."""

    model_config = ConfigDict(frozen=True)

    type: Literal["error"] = "error"
    thread_id: str
    emitted_at: float
    message: str
    retryable: bool
    reason: (
        Literal[
            "rate_limited",
            "budget_exhausted",
            "provider_exhausted",
            "input_too_long",
            "internal_error",
        ]
        | None
    ) = None


GraphEvent = (
    GraphStartedEvent
    | NodeStartedEvent
    | NodeCompletedEvent
    | RetryLoopEvent
    | GraphInterruptedEvent
    | GraphCompletedEvent
    | GraphErrorEvent
)


def planner_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "rewritten_query": result.get("rewritten_query"),
        "intent": result.get("intent"),
        "retry_budget": result.get("retry_budget"),
    }
    if result.get("intent") == QueryIntent.OUT_OF_SCOPE:
        payload["abstained"] = result.get("abstained", True)
        payload["abstain_reason"] = result.get("abstain_reason")
    return payload


def retriever_payload(result: dict[str, Any]) -> dict[str, Any]:
    chunks: list[ScoredChunk] = result.get("retrieved_chunks", [])
    return {"candidates": [c.model_dump() for c in chunks]}


def reranker_payload(result: dict[str, Any]) -> dict[str, Any]:
    chunks: list[ScoredChunk] = result.get("reranked_chunks", [])
    return {"reranked": [c.model_dump() for c in chunks]}


def generator_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "confidence": result.get("confidence"),
        "abstained": result.get("abstained"),
        "abstain_reason": result.get("abstain_reason"),
    }


def critic_payload(result: dict[str, Any]) -> dict[str, Any]:
    needs_retry = result.get("needs_retry", False)
    return {"faithful": not needs_retry, "reason": result.get("critic_feedback")}


def hitl_gate_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {"human_approved": result.get("human_approved")}


NODE_PAYLOAD_BUILDERS: dict[NodeName, Any] = {
    NodeName.PLANNER: planner_payload,
    NodeName.RETRIEVER: retriever_payload,
    NodeName.RERANKER: reranker_payload,
    NodeName.GENERATOR: generator_payload,
    NodeName.CRITIC: critic_payload,
    NodeName.HITL_GATE: hitl_gate_payload,
}
