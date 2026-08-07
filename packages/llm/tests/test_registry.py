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


def test_reason_chain_is_deepseek_first_then_groq_fallback() -> None:
    # See docs/adr/0005-deepseek-primary-groq-free-fallback.md — DeepSeek is
    # primary (no Gemini/Groq-paid yet), Groq's free tier is the fallback. Update
    # this test alongside any further registry change (Groq paid, Gemini).
    registry = build_default_registry(GatewaySettings())
    chain = registry.fallback_chain(Tier.REASON)
    assert chain[0].provider == "deepseek"
    assert all(provider.provider == "groq" for provider in chain[1:])
    assert len(chain) >= 2  # still a real fallback chain, not a single point of failure


def test_reason_chain_is_deepseek_first_in_demo_too() -> None:
    registry = build_default_registry(GatewaySettings(app_env="demo"))
    chain = registry.fallback_chain(Tier.REASON)
    assert chain[0].provider == "deepseek"


def test_fast_and_bulk_chains_are_deepseek_first_then_groq_fallback() -> None:
    registry = build_default_registry(GatewaySettings())
    for tier in (Tier.FAST, Tier.BULK):
        chain = registry.fallback_chain(tier)
        assert chain[0].provider == "deepseek"
        assert all(provider.provider == "groq" for provider in chain[1:])


def test_only_gemini_providers_get_a_daily_request_ceiling() -> None:
    registry = build_default_registry(GatewaySettings())
    for tier in Tier:
        for provider in registry.fallback_chain(tier):
            if provider.provider == "gemini":
                assert provider.daily_request_ceiling is not None
            else:
                assert provider.daily_request_ceiling is None
