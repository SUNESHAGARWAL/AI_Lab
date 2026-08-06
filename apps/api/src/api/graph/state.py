from typing import TypedDict

from core.models import Filters, ScoredChunk

from api.graph.schemas import QueryIntent

DEFAULT_MAX_RETRIES = 2


class AgentState(TypedDict):
    """Single typed state for the whole graph, per CLAUDE.md's graph conventions.
    Nodes read and write this; they never call each other directly — the runtime
    routes between them. Every field is overwritten wholesale by the node that owns
    it (a retry means a fresh retrieval, not accumulation), so no field needs an
    Annotated reducer."""

    query: str
    rewritten_query: str | None
    intent: QueryIntent | None
    filters: Filters | None
    retrieved_chunks: list[ScoredChunk]
    reranked_chunks: list[ScoredChunk]
    answer: str | None
    citations: list[str]
    confidence: float | None
    abstained: bool
    abstain_reason: str | None
    critic_feedback: str | None
    needs_retry: bool
    retry_count: int
    max_retries: int
    retry_budget: int
    human_approved: bool | None


def initial_state(
    query: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    filters: Filters | None = None,
    retry_budget: int | None = None,
) -> AgentState:
    return AgentState(
        query=query,
        rewritten_query=None,
        intent=None,
        filters=filters,
        retrieved_chunks=[],
        reranked_chunks=[],
        answer=None,
        citations=[],
        confidence=None,
        abstained=False,
        abstain_reason=None,
        critic_feedback=None,
        needs_retry=False,
        retry_count=0,
        max_retries=max_retries,
        retry_budget=retry_budget if retry_budget is not None else max_retries,
        human_approved=None,
    )
