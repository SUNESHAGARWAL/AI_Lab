from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from api.graph.nodes import (
    critic,
    generator,
    hitl_gate,
    planner,
    reranker,
    retriever,
    route_after_critic,
)
from api.graph.state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Wires the topology; the checkpointer is injected, not constructed here — same
    dependency-injection shape as packages/llm.Gateway's constructor-supplied Redis
    client."""
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("reranker", reranker)
    builder.add_node("generator", generator)
    builder.add_node("critic", critic)
    builder.add_node("hitl_gate", hitl_gate)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "reranker")
    builder.add_edge("reranker", "generator")
    builder.add_edge("generator", "critic")
    builder.add_conditional_edges(
        "critic", route_after_critic, {"retry": "retriever", "proceed": "hitl_gate"}
    )
    builder.add_edge("hitl_gate", END)

    return builder.compile(checkpointer=checkpointer)
