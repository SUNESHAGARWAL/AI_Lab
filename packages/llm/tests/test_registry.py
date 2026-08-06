from llm.config import GatewaySettings
from llm.models import Tier
from llm.registry import build_default_registry


def test_fallback_chain_is_ordered_and_nonempty() -> None:
    registry = build_default_registry(GatewaySettings())

    for tier in Tier:
        chain = registry.fallback_chain(tier)
        assert len(chain) >= 1
        assert all(provider.model for provider in chain)


def test_local_tier_chain_has_no_network_provider() -> None:
    registry = build_default_registry(GatewaySettings())
    chain = registry.fallback_chain(Tier.LOCAL)
    assert all(provider.provider == "ollama" for provider in chain)


def test_provider_concurrency_override_applies() -> None:
    settings = GatewaySettings(provider_concurrency_overrides={"groq": 99})
    registry = build_default_registry(settings)

    fast_chain = registry.fallback_chain(Tier.FAST)
    groq_providers = [p for p in fast_chain if p.provider == "groq"]
    assert groq_providers
    assert all(p.max_concurrency == 99 for p in groq_providers)


def test_reason_chain_is_groq_first_by_default() -> None:
    registry = build_default_registry(GatewaySettings())
    chain = registry.fallback_chain(Tier.REASON)
    assert chain[0].provider == "groq"
    assert chain[1].provider == "gemini"


def test_reason_chain_is_gemini_first_in_demo() -> None:
    registry = build_default_registry(GatewaySettings(app_env="demo"))
    chain = registry.fallback_chain(Tier.REASON)
    assert chain[0].provider == "gemini"
    assert chain[1].provider == "groq"


def test_only_gemini_providers_get_a_daily_request_ceiling() -> None:
    registry = build_default_registry(GatewaySettings())
    for tier in Tier:
        for provider in registry.fallback_chain(tier):
            if provider.provider == "gemini":
                assert provider.daily_request_ceiling is not None
            else:
                assert provider.daily_request_ceiling is None
