"""Graph nodes — all real logic now. Nodes never call each other directly — the graph
in api.graph.build owns all routing."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from core.models import Query, ScoredChunk
from core.ports import Reranker, Retriever
from langgraph.types import interrupt

from api.graph.schemas import CriticVerdict, GeneratedAnswer, PlannerDecision, QueryIntent
from api.graph.state import AgentState
from llm import CompletionRequest, Gateway, Message, Tier


def _planner_system_prompt(max_retries: int) -> str:
    return (
        "You are a query planner for a citation-grounded question-answering system. "
        "Given the user's raw question, do three things: "
        "1) Rewrite it into a version optimized for retrieval — expand vague "
        "phrasing, resolve pronouns and ambiguous references using context from the "
        "question itself, and add relevant terminology if the intent is clear. If "
        "the question is already clear and well-formed, the rewrite can match the "
        "original. "
        "2) Classify the intent as exactly one of: `factual_lookup` (a direct "
        "factual question answerable from a corpus), `role_scoped_applicability` (a "
        "question about whether or how something applies to a specific role, "
        "situation, or context), or `out_of_scope` (not answerable by this system at "
        "all — off-topic, harmful, or nonsensical). "
        f"3) Set retry_budget to an integer between 0 and {max_retries}: how many "
        "critic-driven regeneration attempts this query is worth — 0 or 1 for a "
        f"simple, unambiguous factual lookup, up to {max_retries} for a complex "
        "multi-part or nuanced question. "
        "If intent is `out_of_scope`, briefly explain why in `abstain_reason`."
    )


def _build_planner_messages(query: str, max_retries: int) -> list[Message]:
    return [
        Message(role="system", content=_planner_system_prompt(max_retries)),
        Message(role="user", content=query),
    ]


def make_planner_node(
    gateway: Gateway, tier: Tier = Tier.FAST
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Binds an llm.Gateway into a graph node — same build-time DI shape as the other
    LLM-backed nodes. Fast tier, cheap: a short rewrite + classification + budget
    call, not a generation pass. See api.graph.schemas.PlannerDecision for the
    response contract. An out-of-scope classification sets abstained=True directly
    here — api.graph.build routes straight to hitl_gate on that, skipping retrieval
    and generation entirely."""

    async def planner_node(state: AgentState) -> dict[str, Any]:
        request = CompletionRequest(
            tier=tier,
            messages=_build_planner_messages(state["query"], state["max_retries"]),
            response_model=PlannerDecision,
            max_tokens=300,
        )
        result = await gateway.complete(request)
        if not isinstance(result.parsed, PlannerDecision):
            raise TypeError(
                "expected a PlannerDecision from gateway.complete() with "
                "response_model set — this indicates a Gateway contract violation"
            )

        parsed = result.parsed
        retry_budget = max(0, min(parsed.retry_budget, state["max_retries"]))

        update: dict[str, Any] = {
            "rewritten_query": parsed.rewritten_query,
            "intent": parsed.intent,
            "retry_budget": retry_budget,
        }
        if parsed.intent is QueryIntent.OUT_OF_SCOPE:
            update.update(
                {
                    "abstained": True,
                    "abstain_reason": (
                        parsed.abstain_reason or "query is out of scope for this system"
                    ),
                    "answer": "",
                    "citations": [],
                    "confidence": 0.0,
                }
            )
        return update

    return planner_node


def route_after_planner(state: AgentState) -> Literal["continue", "abstain"]:
    return "abstain" if state["abstained"] else "continue"


def make_retriever_node(
    retriever: Retriever,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Binds a Retriever port implementation into a graph node — same build-time DI
    shape already used for the checkpointer in api.graph.build. Swapping FakeRetriever
    for a real adapter later means a different argument to build_graph, not a change
    here."""

    async def retriever_node(state: AgentState) -> dict[str, Any]:
        query_text = state["rewritten_query"] or state["query"]
        query = Query(text=query_text, filters=state["filters"])
        results = await retriever.retrieve(query)
        return {"retrieved_chunks": results}

    return retriever_node


DEFAULT_RERANK_TOP_N = 5


def make_reranker_node(
    reranker: Reranker, top_n: int = DEFAULT_RERANK_TOP_N
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Binds a Reranker port implementation into a graph node — same build-time DI
    shape as make_retriever_node. The Reranker port always returns the same set of
    candidates it was given (rescored/reordered, never truncated — see
    core.ports.Reranker's docstring); truncating to top_n is this node's job."""

    async def reranker_node(state: AgentState) -> dict[str, Any]:
        query_text = state["rewritten_query"] or state["query"]
        query = Query(text=query_text, filters=state["filters"])
        reranked = await reranker.rerank(query, state["retrieved_chunks"])
        return {"reranked_chunks": reranked[:top_n]}

    return reranker_node


_SYSTEM_PROMPT = (
    "You are a citation-grounded question-answering assistant. Answer the user's "
    "question using ONLY the information in the chunks provided below — never rely "
    "on outside knowledge, even if you believe you already know the answer. Every "
    "claim in your answer must be traceable to at least one provided chunk; list the "
    "id of every chunk you relied on in `citations`, using only the chunk ids given "
    "below — never invent or guess an id. If the provided chunks do not contain "
    "enough information to answer confidently, set `abstained` to true and explain "
    "why in `abstain_reason` instead of guessing."
)


def _format_chunks(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[chunk_id={sc.chunk.id}]\n{sc.chunk.text}" for sc in chunks)


def _build_messages(query: str, chunks: list[ScoredChunk]) -> list[Message]:
    return [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(
            role="user", content=f"Question: {query}\n\nChunks:\n{_format_chunks(chunks)}"
        ),
    ]


def _validate_citations(parsed: GeneratedAnswer, valid_chunk_ids: set[str]) -> GeneratedAnswer:
    """Defense-in-depth: the prompt instructs the model to cite only provided chunk
    ids, but a model can still hallucinate one despite instructions. A citation to a
    chunk that doesn't exist makes the whole answer untrustworthy, not just that one
    claim — so an unknown id forces abstention rather than silently keeping a
    partially-grounded answer."""
    unknown = [cid for cid in parsed.citations if cid not in valid_chunk_ids]
    if unknown and not parsed.abstained:
        return parsed.model_copy(
            update={
                "abstained": True,
                "abstain_reason": (
                    f"model cited chunk id(s) not present in the input: {unknown}"
                ),
                "confidence": 0.0,
            }
        )
    return parsed


def make_generator_node(
    gateway: Gateway, tier: Tier = Tier.REASON
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Binds an llm.Gateway into a graph node — same build-time DI shape as
    make_retriever_node/make_reranker_node. Calls the reason tier for structured,
    citation-grounded generation; see api.graph.schemas.GeneratedAnswer for the
    response contract this enforces."""

    async def generator_node(state: AgentState) -> dict[str, Any]:
        chunks = state["reranked_chunks"]
        if not chunks:
            # Nothing to ground an answer in — abstain without risking the model
            # answering from outside knowledge despite the prompt.
            return {
                "answer": "",
                "citations": [],
                "confidence": 0.0,
                "abstained": True,
                "abstain_reason": "no chunks were retrieved to ground an answer in",
            }

        query_text = state["rewritten_query"] or state["query"]
        request = CompletionRequest(
            tier=tier,
            messages=_build_messages(query_text, chunks),
            response_model=GeneratedAnswer,
        )
        result = await gateway.complete(request)
        if not isinstance(result.parsed, GeneratedAnswer):
            raise TypeError(
                "expected a GeneratedAnswer from gateway.complete() with "
                "response_model set — this indicates a Gateway contract violation"
            )

        parsed = _validate_citations(result.parsed, {sc.chunk.id for sc in chunks})
        return {
            "answer": parsed.answer,
            "citations": parsed.citations,
            "confidence": parsed.confidence,
            "abstained": parsed.abstained,
            "abstain_reason": parsed.abstain_reason,
        }

    return generator_node


_CRITIC_SYSTEM_PROMPT = (
    "You are a strict fact-checker for a citation-grounded QA system. You will be "
    "given an answer and the specific source chunks it claims to cite. Determine "
    "whether every claim in the answer is actually supported by those chunks — not "
    "just that the citation ids are valid, but that the cited text actually backs up "
    "what's claimed. Set `faithful` to true only if every claim follows directly from "
    "the cited chunks. If any claim is not supported, set `faithful` to false and give "
    "a brief one-sentence `reason` identifying the unsupported claim and which chunk "
    "id it relates to."
)


def _cited_chunks(citations: list[str], chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    citation_ids = set(citations)
    return [sc for sc in chunks if sc.chunk.id in citation_ids]


def _build_critic_messages(
    answer: str, citations: list[str], chunks: list[ScoredChunk]
) -> list[Message]:
    cited = _cited_chunks(citations, chunks)
    return [
        Message(role="system", content=_CRITIC_SYSTEM_PROMPT),
        Message(
            role="user", content=f"Answer: {answer}\n\nCited chunks:\n{_format_chunks(cited)}"
        ),
    ]


def make_critic_node(
    gateway: Gateway, tier: Tier = Tier.FAST
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Binds an llm.Gateway into a graph node — same build-time DI shape as
    make_generator_node, different tier. A lightweight faithfulness self-check (fast
    tier, only the cited chunks shown to the judge, small max_tokens), not a second
    full generation pass — checks that the cited text actually supports the answer's
    claims, not just that the citation ids are valid (make_generator_node's
    _validate_citations already covers that part)."""

    async def critic_node(state: AgentState) -> dict[str, Any]:
        if state["abstained"]:
            # Nothing to fact-check in an honest abstention — and forcing a retry
            # here would pressure the model into fabricating an answer just to
            # satisfy the critic, undermining the abstention behavior itself.
            return {"needs_retry": False, "critic_feedback": None}

        answer = state["answer"] or ""
        request = CompletionRequest(
            tier=tier,
            messages=_build_critic_messages(answer, state["citations"], state["reranked_chunks"]),
            response_model=CriticVerdict,
            max_tokens=200,
        )
        result = await gateway.complete(request)
        if not isinstance(result.parsed, CriticVerdict):
            raise TypeError(
                "expected a CriticVerdict from gateway.complete() with "
                "response_model set — this indicates a Gateway contract violation"
            )

        if result.parsed.faithful:
            return {"needs_retry": False, "critic_feedback": None}

        return {
            "needs_retry": True,
            "retry_count": state["retry_count"] + 1,
            "critic_feedback": result.parsed.reason,
        }

    return critic_node


def route_after_critic(state: AgentState) -> Literal["retry", "proceed"]:
    # max_retries is the absolute hard ceiling, never bypassable; retry_budget is the
    # planner's per-query sub-cap beneath it (defaults to max_retries when unset, so
    # this is identical to a flat max_retries check unless the planner lowered it).
    effective_cap = min(state["retry_budget"], state["max_retries"])
    if state["needs_retry"] and state["retry_count"] < effective_cap:
        return "retry"
    return "proceed"


DEFAULT_CONFIDENCE_THRESHOLD = 0.7


def _needs_human_review(state: AgentState, confidence_threshold: float) -> bool:
    if state["abstained"]:
        return True
    confidence = state["confidence"]
    # Missing confidence fails safe toward triggering review, not skipping it — a
    # node that didn't set confidence is a bug, not a green light.
    return confidence is None or confidence < confidence_threshold


def _apply_hitl_decision(decision: Any) -> dict[str, Any]:
    """Resume contract: a bare bool (True/False -> accept/reject) or a dict
    {"decision": "accept" | "reject" | "edit", ...}. "edit" may carry "answer"/
    "citations" to overwrite the draft. Anything unrecognized falls back to reject —
    fail toward not auto-approving on a malformed resume payload."""
    if isinstance(decision, dict):
        outcome = decision.get("decision")
    else:
        outcome = "accept" if decision else "reject"

    if outcome == "accept":
        return {"human_approved": True}
    if outcome == "edit" and isinstance(decision, dict):
        update: dict[str, Any] = {"human_approved": True}
        if "answer" in decision:
            update["answer"] = decision["answer"]
        if "citations" in decision:
            update["citations"] = decision["citations"]
        return update
    return {"human_approved": False}


def make_hitl_gate_node(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """No LLM gateway or core.ports adapter involved — just langgraph.types.interrupt,
    the only human-in-the-loop mechanism this project uses (ADR 0001). A confident,
    non-abstained answer skips the interrupt entirely and passes straight through in
    the same ainvoke() call. Otherwise the payload shape depends on why a human is
    being asked to look: an out-of-scope query (no draft to show) gets a rephrase/
    contact-a-human framing; everything else gets the draft answer and citations to
    accept, reject, or edit."""

    async def hitl_gate_node(state: AgentState) -> dict[str, Any]:
        if not _needs_human_review(state, confidence_threshold):
            return {"human_approved": True}  # auto-approved: no review was needed

        if state["intent"] is QueryIntent.OUT_OF_SCOPE:
            payload = {
                "type": "out_of_scope",
                "message": "This question is outside what this system can help with.",
                "reason": state["abstain_reason"],
                "suggestion": "Try rephrasing your question, or contact a human for help.",
            }
        else:
            payload = {
                "type": "review",
                "answer": state["answer"],
                "citations": state["citations"],
                "confidence": state["confidence"],
                "abstained": state["abstained"],
                "abstain_reason": state["abstain_reason"],
            }

        return _apply_hitl_decision(interrupt(payload))

    return hitl_gate_node
