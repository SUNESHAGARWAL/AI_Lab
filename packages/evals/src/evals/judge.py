"""LLM-judge plumbing for Layer 3, all routed through llm.Gateway — see
docs/adr/0004-generation-eval-judge-via-deepeval-not-ragas.md for why deepeval's
DeepEvalBaseLLM (this module's GatewayJudgeModel) is used instead of ragas's
langchain-coupled BaseRagasLLM."""

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel, ConfigDict

from llm import CompletionRequest, Gateway, Message, Tier, complete_json

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
    """Our own structured judge call — via llm.complete_json (plain-text completion
    + robust parsing), NOT CompletionRequest.response_model. Groq's small models
    (both this project's registry entries, per litellm.supports_response_schema)
    don't support native structured output; requesting it forces LiteLLM into a
    tool-calling workaround these models invoke unreliably, surfacing as Groq's
    tool_use_failed error. complete_json sidesteps that entirely — see
    packages/llm/src/llm/prompted_json.py."""
    messages = [
        Message(role="system", content=_CITATION_SUPPORT_SYSTEM_PROMPT),
        Message(
            role="user",
            content=f"Answer:\n{answer}\n\nCited chunk:\n{chunk_text}",
        ),
    ]
    # 350, not a tighter number: this tier's Groq fallback chain includes a
    # gpt-oss model, which spends part of its budget on an internal reasoning pass
    # before visible content — verified empirically that a too-tight max_tokens can
    # return empty content on that model. See llm/registry.py's docstring.
    return await complete_json(gateway, tier, messages, CitationSupportVerdict, max_tokens=350)
