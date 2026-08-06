"""Model tiers and fallback chains — the single place a model string may appear.
Nodes ask for a `Tier`, never a model; see `.claude/rules/llm-gateway.md`.

The default chains below are placeholders representative of free-tier providers
(Groq, Gemini, local Ollama) at time of writing. Free-tier availability and quotas
shift roughly monthly — verify current offerings before relying on this in
production, the same caveat ADR 0001 raised about its own version pins. Ceilings
(concurrency, RPM) are overridable via `GatewaySettings` without touching this file.
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


_GROQ_REASON = ProviderModel(
    provider="groq", model="groq/llama-3.3-70b-versatile", max_concurrency=4
)
_GEMINI_REASON = ProviderModel(
    provider="gemini", model="gemini/gemini-2.0-flash", max_concurrency=4
)


def _reason_chain(app_env: str) -> list[ProviderModel]:
    """Gemini-first only in demo/production, where its large context window is the
    point of this tier. Everywhere else (development, or anything unrecognized) tries
    Groq first, so local dev iteration and smoke tests don't touch Gemini's daily
    quota unless Groq's chain is actually exhausted."""
    if app_env == "demo":
        return [_GEMINI_REASON, _GROQ_REASON]
    return [_GROQ_REASON, _GEMINI_REASON]


_STATIC_CHAINS: dict[Tier, list[ProviderModel]] = {
    Tier.FAST: [
        ProviderModel(provider="groq", model="groq/llama-3.1-8b-instant", max_concurrency=8),
        ProviderModel(
            provider="gemini", model="gemini/gemini-2.0-flash-lite", max_concurrency=4
        ),
    ],
    Tier.BULK: [
        ProviderModel(provider="gemini", model="gemini/gemini-2.0-flash", max_concurrency=4),
        ProviderModel(provider="groq", model="groq/llama-3.1-8b-instant", max_concurrency=8),
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
