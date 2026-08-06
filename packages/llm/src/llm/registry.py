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


_DEFAULT_CHAINS: dict[Tier, list[ProviderModel]] = {
    Tier.FAST: [
        ProviderModel(provider="groq", model="groq/llama-3.1-8b-instant", max_concurrency=8),
        ProviderModel(
            provider="gemini", model="gemini/gemini-2.0-flash-lite", max_concurrency=4
        ),
    ],
    Tier.REASON: [
        ProviderModel(provider="gemini", model="gemini/gemini-2.0-flash", max_concurrency=4),
        ProviderModel(
            provider="groq", model="groq/llama-3.3-70b-versatile", max_concurrency=4
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
    chains: dict[Tier, list[ProviderModel]] = {}
    for tier, providers in _DEFAULT_CHAINS.items():
        chains[tier] = [
            provider.model_copy(
                update={
                    "max_concurrency": settings.provider_concurrency_overrides.get(
                        provider.provider, provider.max_concurrency
                    )
                }
            )
            for provider in providers
        ]
    return TierRegistry(chains)
