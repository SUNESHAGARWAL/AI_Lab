"""Graph nodes — stub bodies only. Each returns a valid partial-state update with
correct typing; no real retrieval/generation/critique logic yet. That lands with
packages/retrieval and packages/llm.Gateway wiring, as separate work. Nodes never
call each other directly — the graph in api.graph.build owns all routing."""

from typing import Any, Literal

from langgraph.types import interrupt

from api.graph.state import AgentState


async def planner(state: AgentState) -> dict[str, Any]:
    """Query rewrite / disambiguation. Stub: passes the query through unchanged."""
    return {"rewritten_query": state["query"]}


async def retriever(state: AgentState) -> dict[str, Any]:
    """Stub: no real retrieval yet — packages/retrieval's Retriever port lands
    separately."""
    return {"retrieved_chunks": []}


async def reranker(state: AgentState) -> dict[str, Any]:
    """Stub: passes retrieved chunks through unranked."""
    return {"reranked_chunks": state["retrieved_chunks"]}


async def generator(state: AgentState) -> dict[str, Any]:
    """Stub: no real packages/llm.Gateway call yet."""
    return {"answer": "stub answer", "citations": []}


async def critic(state: AgentState) -> dict[str, Any]:
    """Stub: unconditionally requests a retry — there's no real judgment yet, so
    "always wants another pass" is the honest stub behavior. The hard cap in
    route_after_critic is what actually stops the loop; that's the property this
    scaffold needs to prove works, not critic quality."""
    return {
        "needs_retry": True,
        "retry_count": state["retry_count"] + 1,
        "critic_feedback": "stub: not yet evaluated",
    }


def route_after_critic(state: AgentState) -> Literal["retry", "proceed"]:
    if state["needs_retry"] and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "proceed"


async def hitl_gate(state: AgentState) -> dict[str, Any]:
    """Stub: pauses for human approval of the final answer via interrupt() — the
    only human-in-the-loop mechanism this project uses, per ADR 0001."""
    decision = interrupt({"answer": state["answer"], "citations": state["citations"]})
    return {"human_approved": bool(decision)}
