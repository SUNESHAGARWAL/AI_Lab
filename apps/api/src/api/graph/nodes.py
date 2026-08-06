"""Graph nodes. planner is still a stub; retriever/reranker/generator/critic have real
logic. Nodes never call each other directly — the graph in api.graph.build owns all
routing."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from core.models import Query, ScoredChunk
from core.ports import Reranker, Retriever
from langgraph.types import interrupt

from api.graph.schemas import CriticVerdict, GeneratedAnswer
from api.graph.state import AgentState
from llm import CompletionRequest, Gateway, Message, Tier


async def planner(state: AgentState) -> dict[str, Any]:
    """Query rewrite / disambiguation. Stub: passes the query through unchanged."""
    return {"rewritten_query": state["query"]}


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
    if state["needs_retry"] and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "proceed"


async def hitl_gate(state: AgentState) -> dict[str, Any]:
    """Stub: pauses for human approval of the final answer via interrupt() — the
    only human-in-the-loop mechanism this project uses, per ADR 0001."""
    decision = interrupt({"answer": state["answer"], "citations": state["citations"]})
    return {"human_approved": bool(decision)}
