"""Model tiers and fallback chains — the single place a model string may appear.
Nodes ask for a `Tier`, never a model; see `.claude/rules/llm-gateway.md`.

The default chains below are placeholders representative of free-tier providers
(Groq, Gemini, local Ollama) at time of writing. Free-tier availability and quotas
shift roughly monthly — verify current offerings before relying on this in
production, the same caveat ADR 0001 raised about its own version pins. Ceilings
(concurrency, RPM) are overridable via `GatewaySettings` without touching this file.

Current state (see docs/adr/0005-deepseek-primary-groq-free-fallback.md for the
full record): **DeepSeek is the primary provider for every network tier, Groq's
free tier is the fallback.** This followed diagnosing repeated Groq 429s down to a
real, current tokens-per-minute (TPM) limit — not the daily quota, which resets
every 24h and was never close to exhausted. Reproduced directly against Groq's API:
`"tokens per minute (TPM): Limit 6000, Used 3038, Requested 3080."` on
`llama-3.1-8b-instant`; confirmed TPM ceilings of 6,000-12,000 across every Groq
model in this registry — tight against this project's real prompt sizes (full AI
Act/GDPR article chunks). DeepSeek's free tier is a 500K-token/**day** allowance
with no per-minute wall, comfortably covers this project's full usage estimate for
$0, and (confirmed via `litellm.supports_response_schema`) supports *native*
structured output — unlike Groq's small models, which make LiteLLM fall back to a
tool-calling workaround these models invoke unreliably (see
`packages/llm/src/llm/prompted_json.py`'s docstring for the `tool_use_failed`
consequence of that).

Groq's *paid* Dev Tier isn't set up yet — add it as a fallback ahead of the free
entries once it is (removes the TPM ceiling entirely for the fallback path too).
Gemini stays disabled: every `_GEMINI_*` ProviderModel and the demo-vs-dev
Gemini-first branch in `_reason_chain` are commented out, not deleted, pending both
an account/quota fix and switching off the now-deprecated `gemini-2.0-flash`/
`gemini-2.0-flash-lite` model ids to `gemini-2.5-flash-lite` or newer.
"""

from pydantic import BaseModel, ConfigDict

from llm.config import GatewaySettings
from llm.models import Tier


class ProviderModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    max_concurrency: int = 4
    # Soft daily-request ceiling for this provider, or None if untracked. Only
    # Gemini's entries get one populated (see build_default_registry) — the field
    # itself stays generic so the skip mechanism in gateway.py isn't a
    # provider-name special case.
    daily_request_ceiling: int | None = None


_DEEPSEEK_CHAT = ProviderModel(
    provider="deepseek", model="deepseek/deepseek-chat", max_concurrency=8
)
_DEEPSEEK_REASON = ProviderModel(
    provider="deepseek", model="deepseek/deepseek-reasoner", max_concurrency=4
)
_GROQ_REASON = ProviderModel(
    provider="groq", model="groq/llama-3.3-70b-versatile", max_concurrency=4
)
_GROQ_REASON_FALLBACK = ProviderModel(
    provider="groq", model="groq/openai/gpt-oss-120b", max_concurrency=4
)
# _GEMINI_REASON = ProviderModel(
#     provider="gemini", model="gemini/gemini-2.0-flash", max_concurrency=4
# )


def _reason_chain(app_env: str) -> list[ProviderModel]:
    """DeepSeek primary, Groq free tier fallback — see the module docstring. The
    app_env parameter and demo-vs-dev branch are kept (not removed) so restoring
    Gemini's original "demo/production prefers Gemini's larger context window"
    behavior is a matter of un-commenting, not re-deriving this function."""
    # if app_env == "demo":
    #     return [_GEMINI_REASON, _DEEPSEEK_REASON, _GROQ_REASON, _GROQ_REASON_FALLBACK]
    del app_env  # unused while Gemini is disabled — see above
    return [_DEEPSEEK_REASON, _GROQ_REASON, _GROQ_REASON_FALLBACK]


_STATIC_CHAINS: dict[Tier, list[ProviderModel]] = {
    Tier.FAST: [
        _DEEPSEEK_CHAT,
        ProviderModel(provider="groq", model="groq/llama-3.1-8b-instant", max_concurrency=8),
        ProviderModel(provider="groq", model="groq/openai/gpt-oss-20b", max_concurrency=8),
        # ProviderModel(
        #     provider="gemini", model="gemini/gemini-2.0-flash-lite", max_concurrency=4
        # ),
    ],
    Tier.BULK: [
        _DEEPSEEK_CHAT,
        ProviderModel(provider="groq", model="groq/llama-3.1-8b-instant", max_concurrency=8),
        ProviderModel(provider="groq", model="groq/openai/gpt-oss-20b", max_concurrency=8),
        # ProviderModel(provider="gemini", model="gemini/gemini-2.0-flash", max_concurrency=4),
    ],
    Tier.LOCAL: [
        ProviderModel(provider="ollama", model="ollama/llama3.1", max_concurrency=2),
    ],
}


class TierRegistry:
    def __init__(self, chains: dict[Tier, list[ProviderModel]]) -> None:
        self._chains = chains

    def fallback_chain(self, tier: Tier) -> list[ProviderModel]:
        return self._chains[tier]


def build_default_registry(settings: GatewaySettings) -> TierRegistry:
    default_chains: dict[Tier, list[ProviderModel]] = {
        **_STATIC_CHAINS,
        Tier.REASON: _reason_chain(settings.app_env),
    }
    chains: dict[Tier, list[ProviderModel]] = {}
    for tier, providers in default_chains.items():
        chains[tier] = [
            provider.model_copy(
                update={
                    "max_concurrency": settings.provider_concurrency_overrides.get(
                        provider.provider, provider.max_concurrency
                    ),
                    "daily_request_ceiling": (
                        settings.gemini_daily_request_ceiling
                        if provider.provider == "gemini"
                        else None
                    ),
                }
            )
            for provider in providers
        ]
    return TierRegistry(chains)
