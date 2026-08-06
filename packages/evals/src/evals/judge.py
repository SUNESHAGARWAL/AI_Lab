"""LLM-judge plumbing for Layer 3, all routed through llm.Gateway — see
docs/adr/0004-generation-eval-judge-via-deepeval-not-ragas.md for why deepeval's
DeepEvalBaseLLM (this module's GatewayJudgeModel) is used instead of ragas's
langchain-coupled BaseRagasLLM."""

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel, ConfigDict

from llm import CompletionRequest, Gateway, Message, Tier

DEFAULT_JUDGE_TIER = Tier.FAST


class GatewayJudgeModel(DeepEvalBaseLLM):
    """Routes every deepeval judge call through llm.Gateway — retries, budget guard,
    exact-match caching (keyed on tier/messages/temperature/max_tokens — see
    llm.cache), and telemetry are all inherited for free. deepeval's async_mode
    (the default for every metric this package uses) only ever calls a_generate."""

    def __init__(self, gateway: Gateway, tier: Tier = DEFAULT_JUDGE_TIER) -> None:
        self._gateway = gateway
        self._tier = tier
        super().__init__(model=f"gateway:{tier.value}")

    def load_model(self) -> "GatewayJudgeModel":
        return self

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "GatewayJudgeModel only supports async judging — deepeval's metrics are "
            "always constructed/measured with async_mode=True in this package"
        )

    async def a_generate(self, prompt: str) -> str:
        request = CompletionRequest(
            tier=self._tier,
            messages=[Message(role="user", content=prompt)],
            max_tokens=2048,
        )
        result = await self._gateway.complete(request)
        return result.text

    def get_model_name(self) -> str:
        return f"gateway:{self._tier.value}"


_CITATION_SUPPORT_SYSTEM_PROMPT = (
    "You are judging whether a single source chunk actually supports an answer that "
    "cites it. Read the chunk and the answer. Set `supports` to true only if the "
    "chunk provides real evidence for at least part of what the answer claims — not "
    "just topical overlap. Set it to false if the chunk is unrelated or only "
    "tangentially related to the answer's claims. Give a brief one-sentence `reason`."
)


class CitationSupportVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    supports: bool
    reason: str | None = None


async def judge_citation_support(
    gateway: Gateway,
    chunk_text: str,
    answer: str,
    tier: Tier = DEFAULT_JUDGE_TIER,
) -> CitationSupportVerdict:
    """Our own structured-output judge call (Pydantic response_model, not deepeval) —
    same pattern as apps.api.graph.nodes' critic node."""
    request = CompletionRequest(
        tier=tier,
        messages=[
            Message(role="system", content=_CITATION_SUPPORT_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Answer:\n{answer}\n\nCited chunk:\n{chunk_text}",
            ),
        ],
        response_model=CitationSupportVerdict,
        max_tokens=150,
    )
    result = await gateway.complete(request)
    if not isinstance(result.parsed, CitationSupportVerdict):
        raise TypeError(
            "expected a CitationSupportVerdict from gateway.complete() with "
            "response_model set — this indicates a Gateway contract violation"
        )
    return result.parsed
