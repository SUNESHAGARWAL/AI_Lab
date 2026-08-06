from core.ports import Reranker, Retriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from api.graph.nodes import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_RERANK_TOP_N,
    make_critic_node,
    make_generator_node,
    make_hitl_gate_node,
    make_planner_node,
    make_reranker_node,
    make_retriever_node,
    route_after_critic,
    route_after_planner,
)
from api.graph.state import AgentState
from llm import Gateway, Tier


def build_graph(
    checkpointer: BaseCheckpointSaver,
    retriever: Retriever,
    reranker: Reranker,
    gateway: Gateway,
    *,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    planner_tier: Tier = Tier.FAST,
    generation_tier: Tier = Tier.REASON,
    critic_tier: Tier = Tier.FAST,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> CompiledStateGraph:
    """Wires the topology; the checkpointer, retriever, reranker, and gateway are
    injected, not constructed here — same dependency-injection shape as
    packages/llm.Gateway's constructor-supplied Redis client."""
    builder = StateGraph(AgentState)
    builder.add_node("planner", make_planner_node(gateway, tier=planner_tier))
    builder.add_node("retriever", make_retriever_node(retriever))
    builder.add_node("reranker", make_reranker_node(reranker, top_n=rerank_top_n))
    builder.add_node("generator", make_generator_node(gateway, tier=generation_tier))
    builder.add_node("critic", make_critic_node(gateway, tier=critic_tier))
    builder.add_node(
        "hitl_gate", make_hitl_gate_node(confidence_threshold=confidence_threshold)
    )

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner", route_after_planner, {"continue": "retriever", "abstain": "hitl_gate"}
    )
    builder.add_edge("retriever", "reranker")
    builder.add_edge("reranker", "generator")
    builder.add_edge("generator", "critic")
    builder.add_conditional_edges(
        "critic", route_after_critic, {"retry": "retriever", "proceed": "hitl_gate"}
    )
    builder.add_edge("hitl_gate", END)

    return builder.compile(checkpointer=checkpointer)
