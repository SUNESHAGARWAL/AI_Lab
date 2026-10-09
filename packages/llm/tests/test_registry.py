import llm.registry as registry_module
import pytest
from llm.config import GatewaySettings
from llm.models import Tier
from llm.registry import ProviderModel, build_default_registry


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


def test_deepseek_uses_current_model_with_thinking_split_by_tier() -> None:
    # deepseek-chat / deepseek-reasoner are retired names (ADR 0005 addendum). Both
    # were modes of one model; the split now lives in ProviderModel.thinking, and
    # FAST/BULK must turn it off explicitly — deepseek-flash defaults to on.
    registry = build_default_registry(GatewaySettings())
    for tier, thinking in (
        (Tier.FAST, "disabled"),
        (Tier.BULK, "disabled"),
        (Tier.REASON, "enabled"),
    ):
        primary = registry.fallback_chain(tier)[0]
        assert primary.model == "deepseek/deepseek-flash"
        assert primary.thinking == thinking


# Run pin (LLM_AUDIT_PIN_DEEPSEEK). DeepSeek model ids priced by the llm-audit snapshot
# llm-audit/src/llm_audit/pricing/snapshots/litellm-a2bf67a.json, copied here as a
# literal so AI Lab never imports llm-audit. Update both together.
_SNAPSHOT_DEEPSEEK_MODELS = {"deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"}
_HOSTED_TIERS = (Tier.FAST, Tier.REASON, Tier.BULK)


def _settings(
    *,
    audit_pin_deepseek: bool = False,
    app_env: str = "development",
    provider_concurrency_overrides: dict[str, int] | None = None,
) -> GatewaySettings:
    # `_env_file=None` keeps a developer's local `.env` out of these checks.
    return GatewaySettings(  # type: ignore[call-arg]  # pydantic-settings init kwarg
        _env_file=None,
        app_env=app_env,
        audit_pin_deepseek=audit_pin_deepseek,
        provider_concurrency_overrides=provider_concurrency_overrides or {},
    )


def _shape(chain: list[ProviderModel]) -> list[tuple[str, str, str | None]]:
    return [(p.provider, p.model, p.thinking) for p in chain]


def test_unpinned_registry_equals_todays_chains() -> None:
    unpinned = build_default_registry(_settings(audit_pin_deepseek=False))
    default = build_default_registry(_settings())

    expected = {
        Tier.FAST: [
            ("deepseek", "deepseek/deepseek-flash", "disabled"),
            ("groq", "groq/llama-3.1-8b-instant", None),
            ("groq", "groq/openai/gpt-oss-20b", None),
        ],
        Tier.BULK: [
            ("deepseek", "deepseek/deepseek-flash", "disabled"),
            ("groq", "groq/llama-3.1-8b-instant", None),
            ("groq", "groq/openai/gpt-oss-20b", None),
        ],
        Tier.REASON: [
            ("deepseek", "deepseek/deepseek-flash", "enabled"),
            ("groq", "groq/llama-3.3-70b-versatile", None),
            ("groq", "groq/openai/gpt-oss-120b", None),
        ],
        Tier.LOCAL: [("ollama", "ollama/llama3.1", None)],
    }
    for tier in Tier:
        assert unpinned.fallback_chain(tier) == default.fallback_chain(tier)
        assert _shape(unpinned.fallback_chain(tier)) == expected[tier]


def test_pinned_hosted_chains_hold_deepseek_only_with_thinking_kept() -> None:
    registry = build_default_registry(_settings(audit_pin_deepseek=True))

    for tier, thinking in (
        (Tier.FAST, "disabled"),
        (Tier.REASON, "enabled"),
        (Tier.BULK, "disabled"),
    ):
        chain = registry.fallback_chain(tier)
        assert chain
        assert all(p.provider == "deepseek" for p in chain)
        assert all(p.thinking == thinking for p in chain)


def test_pinned_model_ids_are_in_the_price_snapshot() -> None:
    for app_env in ("development", "demo"):
        registry = build_default_registry(_settings(audit_pin_deepseek=True, app_env=app_env))
        for tier in _HOSTED_TIERS:
            for provider in registry.fallback_chain(tier):
                assert provider.model in _SNAPSHOT_DEEPSEEK_MODELS


def test_pinned_chains_keep_concurrency_overrides() -> None:
    registry = build_default_registry(
        _settings(audit_pin_deepseek=True, provider_concurrency_overrides={"deepseek": 3})
    )
    for tier in _HOSTED_TIERS:
        chain = registry.fallback_chain(tier)
        assert chain
        assert all(p.max_concurrency == 3 for p in chain)
        assert all(p.daily_request_ceiling is None for p in chain)


def test_local_chain_is_unchanged_by_the_pin() -> None:
    unpinned = build_default_registry(_settings(audit_pin_deepseek=False))
    pinned = build_default_registry(_settings(audit_pin_deepseek=True))

    assert pinned.fallback_chain(Tier.LOCAL) == unpinned.fallback_chain(Tier.LOCAL)
    assert _shape(pinned.fallback_chain(Tier.LOCAL)) == [("ollama", "ollama/llama3.1", None)]


def test_pin_with_a_chain_without_deepseek_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groq_only = ProviderModel(provider="groq", model="groq/llama-3.1-8b-instant")
    monkeypatch.setitem(registry_module._STATIC_CHAINS, Tier.BULK, [groq_only])

    with pytest.raises(ValueError, match="bulk"):
        build_default_registry(_settings(audit_pin_deepseek=True))
    # The same chains still build with the pin off.
    assert build_default_registry(_settings()).fallback_chain(Tier.BULK) == [groq_only]


def test_pin_with_a_reason_chain_without_deepseek_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groq_only = ProviderModel(provider="groq", model="groq/llama-3.3-70b-versatile")
    monkeypatch.setattr(registry_module, "_reason_chain", lambda app_env: [groq_only])

    with pytest.raises(ValueError, match="reason"):
        build_default_registry(_settings(audit_pin_deepseek=True))
