"""Runs the real, full LangGraph graph (api.graph.build.build_graph) end-to-end per
golden-set item. See docs/adr/0004-generation-eval-judge-via-deepeval-not-ragas.md
for why packages/evals depending on apps/api is a deliberate exception here."""

from api.graph.build import build_graph
from api.graph.state import initial_state
from core.ports import Retriever
from core.testing import FakeReranker
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from evals.golden import Difficulty, GoldenItem, QuestionType
from llm import AllProvidersExhausted, BudgetExceeded, Gateway
from telemetry import get_logger

logger = get_logger(__name__)


class GenerationRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    question_type: QuestionType
    difficulty: Difficulty
    relevant_chunk_ids: list[str]
    answer: str
    citations: list[str]
    confidence: float | None
    abstained: bool
    abstain_reason: str | None
    context_chunk_ids: list[str]
    context_chunks: dict[str, str]


def build_generation_graph(retriever: Retriever, gateway: Gateway) -> CompiledStateGraph:
    """FakeReranker() matches apps/api/src/api/main.py's actual default (see ADR
    0003) — Layer 3 evaluates the graph as it's really deployed, not a reranker
    already decided against. InMemorySaver: no restart-survival needed for a
    one-shot eval run."""
    return build_graph(InMemorySaver(), retriever, FakeReranker(), gateway)


async def run_golden_set_through_graph(
    golden_items: list[GoldenItem], graph: CompiledStateGraph
) -> list[GenerationRunResult]:
    """One ainvoke() per item, its own thread_id. Reads answer/citations/confidence/
    abstained/reranked_chunks straight off the returned state dict regardless of
    whether the run reached END or paused at hitl_gate — hitl_gate only reads those
    fields for its interrupt payload, it doesn't gate their presence, and there is
    no human in this loop to resume the pause.

    A single item's real free-tier provider failure (rate limit exhausting every
    fallback in a tier's chain, or an oversized prompt tripping the per-request
    budget guard) does not abort the whole batch — it's logged loudly and that item
    is skipped, so a transient 429 on item 3 doesn't cost the other 35 results."""
    results: list[GenerationRunResult] = []
    for item in golden_items:
        config = {"configurable": {"thread_id": item.id}}
        try:
            state = await graph.ainvoke(initial_state(item.question), config=config)
        except (AllProvidersExhausted, BudgetExceeded) as exc:
            logger.warning(
                "evals.generation_item_failed",
                id=item.id,
                error=type(exc).__name__,
                message=str(exc),
            )
            continue

        reranked_chunks = state["reranked_chunks"]
        results.append(
            GenerationRunResult(
                id=item.id,
                question=item.question,
                question_type=item.question_type,
                difficulty=item.difficulty,
                relevant_chunk_ids=item.relevant_chunk_ids,
                answer=state["answer"] or "",
                citations=state["citations"],
                confidence=state["confidence"],
                abstained=state["abstained"],
                abstain_reason=state["abstain_reason"],
                context_chunk_ids=[sc.chunk.id for sc in reranked_chunks],
                context_chunks={sc.chunk.id: sc.chunk.text for sc in reranked_chunks},
            )
        )
    return results
