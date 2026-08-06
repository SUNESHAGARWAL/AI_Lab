from typing import TypedDict

from core.models import ScoredChunk

DEFAULT_MAX_RETRIES = 2


class AgentState(TypedDict):
    """Single typed state for the whole graph, per CLAUDE.md's graph conventions.
    Nodes read and write this; they never call each other directly — the runtime
    routes between them. Every field is overwritten wholesale by the node that owns
    it (a retry means a fresh retrieval, not accumulation), so no field needs an
    Annotated reducer."""

    query: str
    rewritten_query: str | None
    retrieved_chunks: list[ScoredChunk]
    reranked_chunks: list[ScoredChunk]
    answer: str | None
    citations: list[str]
    critic_feedback: str | None
    needs_retry: bool
    retry_count: int
    max_retries: int
    human_approved: bool | None


def initial_state(query: str, max_retries: int = DEFAULT_MAX_RETRIES) -> AgentState:
    return AgentState(
        query=query,
        rewritten_query=None,
        retrieved_chunks=[],
        reranked_chunks=[],
        answer=None,
        citations=[],
        critic_feedback=None,
        needs_retry=False,
        retry_count=0,
        max_retries=max_retries,
        human_approved=None,
    )
