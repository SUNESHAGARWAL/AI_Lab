from typing import Any

import httpx
import pytest
from conftest import make_response
from core.testing import FakeEmbedder
from fakeredis import FakeAsyncRedis
from litellm.exceptions import BadRequestError, RateLimitError
from llm.config import GatewaySettings
from llm.errors import AllProvidersExhausted, BudgetExceeded
from llm.gateway import Gateway
from llm.models import CompletionRequest, Message, Tier
from llm.provider_usage import ProviderUsageTracker
from llm.registry import ProviderModel, TierRegistry
from pydantic import BaseModel


def _registry(*providers: ProviderModel) -> TierRegistry:
    return TierRegistry({tier: list(providers) for tier in Tier})


def _rate_limit_error(model: str, retry_after: str = "0") -> RateLimitError:
    return RateLimitError(
        message="rate limited",
        llm_provider="test",
        model=model,
        headers={"retry-after": retry_after},
    )


@pytest.mark.asyncio
async def test_success_path(fake_redis: FakeAsyncRedis) -> None:
    async def completion_fn(**kwargs: Any) -> Any:
        return make_response("hello world")

    provider = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    gateway = Gateway(
        settings=GatewaySettings(),
        registry=_registry(provider),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    result = await gateway.complete(
        CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
    )

    assert result.text == "hello world"
    assert result.cache_hit is False
    assert result.provider == "p1"
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_fallback_moves_to_next_provider_on_rate_limit(fake_redis: FakeAsyncRedis) -> None:
    calls: list[str] = []

    async def completion_fn(**kwargs: Any) -> Any:
        model = kwargs["model"]
        calls.append(model)
        if model == "p1/model":
            raise _rate_limit_error(model)
        return make_response("from p2")

    provider1 = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    provider2 = ProviderModel(provider="p2", model="p2/model", max_concurrency=4)
    gateway = Gateway(
        settings=GatewaySettings(same_provider_retry_attempts=1),
        registry=_registry(provider1, provider2),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    result = await gateway.complete(
        CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
    )

    assert result.provider == "p2"
    assert result.text == "from p2"
    assert result.retry_count >= 1
    assert calls == ["p1/model", "p2/model"]


@pytest.mark.asyncio
async def test_fallback_moves_on_from_mistyped_but_real_429(fake_redis: FakeAsyncRedis) -> None:
    """Regression: a live run against Gemini's free tier surfaced a quota-exceeded
    429 mapped by LiteLLM to `BadRequestError` — a type this gateway doesn't
    otherwise treat as retryable. Without status-code sniffing this propagated raw
    instead of falling back to the next provider."""
    calls: list[str] = []

    async def completion_fn(**kwargs: Any) -> Any:
        model = kwargs["model"]
        calls.append(model)
        if model == "p1/model":
            response = httpx.Response(
                status_code=429, request=httpx.Request("GET", "https://example.com")
            )
            raise BadRequestError(
                message="quota exceeded", model=model, llm_provider="test", response=response
            )
        return make_response("from p2")

    provider1 = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    provider2 = ProviderModel(provider="p2", model="p2/model", max_concurrency=4)
    gateway = Gateway(
        settings=GatewaySettings(same_provider_retry_attempts=1),
        registry=_registry(provider1, provider2),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    result = await gateway.complete(
        CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
    )

    assert result.provider == "p2"
    assert calls == ["p1/model", "p2/model"]


@pytest.mark.asyncio
async def test_genuine_bad_request_propagates_without_fallback(fake_redis: FakeAsyncRedis) -> None:
    """A real 400 (caller bug — malformed request) must NOT be silently retried or
    routed to the next provider; that would mask the bug as a capacity problem."""
    calls: list[str] = []

    async def completion_fn(**kwargs: Any) -> Any:
        calls.append(kwargs["model"])
        response = httpx.Response(
            status_code=400, request=httpx.Request("GET", "https://example.com")
        )
        raise BadRequestError(
            message="malformed request",
            model=kwargs["model"],
            llm_provider="test",
            response=response,
        )

    provider1 = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    provider2 = ProviderModel(provider="p2", model="p2/model", max_concurrency=4)
    gateway = Gateway(
        settings=GatewaySettings(same_provider_retry_attempts=1),
        registry=_registry(provider1, provider2),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    with pytest.raises(BadRequestError):
        await gateway.complete(
            CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
        )

    assert calls == ["p1/model"]  # never rotated to p2


@pytest.mark.asyncio
async def test_soft_daily_limit_skips_gemini_without_attempting_call(
    fake_redis: FakeAsyncRedis,
) -> None:
    calls: list[str] = []

    async def completion_fn(**kwargs: Any) -> Any:
        calls.append(kwargs["model"])
        return make_response("from groq")

    gemini = ProviderModel(
        provider="gemini",
        model="gemini/gemini-2.0-flash",
        max_concurrency=4,
        daily_request_ceiling=10,
    )
    groq = ProviderModel(provider="groq", model="groq/model", max_concurrency=4)
    settings = GatewaySettings(gemini_soft_limit_fraction=0.9)
    gateway = Gateway(
        settings=settings,
        registry=_registry(gemini, groq),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    tracker = ProviderUsageTracker(fake_redis)
    for _ in range(9):  # 9/10 = 90% >= 0.9 fraction -> at the soft limit
        await tracker.increment("gemini")

    result = await gateway.complete(
        CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
    )

    assert calls == ["groq/model"]  # gemini never attempted
    assert result.provider == "groq"


@pytest.mark.asyncio
async def test_gemini_is_attempted_when_below_soft_limit(fake_redis: FakeAsyncRedis) -> None:
    calls: list[str] = []

    async def completion_fn(**kwargs: Any) -> Any:
        calls.append(kwargs["model"])
        return make_response("from gemini")

    gemini = ProviderModel(
        provider="gemini",
        model="gemini/gemini-2.0-flash",
        max_concurrency=4,
        daily_request_ceiling=10,
    )
    settings = GatewaySettings(gemini_soft_limit_fraction=0.9)
    gateway = Gateway(
        settings=settings,
        registry=_registry(gemini),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    tracker = ProviderUsageTracker(fake_redis)
    for _ in range(8):  # 8/10 = 80% < 0.9 fraction -> below the soft limit
        await tracker.increment("gemini")

    result = await gateway.complete(
        CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
    )

    assert calls == ["gemini/gemini-2.0-flash"]
    assert result.provider == "gemini"
    assert await tracker.count("gemini") == 9  # incremented again on this success


@pytest.mark.asyncio
async def test_all_providers_exhausted(fake_redis: FakeAsyncRedis) -> None:
    async def completion_fn(**kwargs: Any) -> Any:
        raise _rate_limit_error(kwargs["model"])

    provider = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    settings = GatewaySettings(same_provider_retry_attempts=1)
    gateway = Gateway(
        settings=settings,
        registry=_registry(provider),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    with pytest.raises(AllProvidersExhausted):
        await gateway.complete(
            CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
        )

    from llm.budget import _day_key

    day_total = await fake_redis.get(_day_key())
    assert day_total is None or int(day_total) == 0


@pytest.mark.asyncio
async def test_budget_guard_blocks_before_calling_provider(fake_redis: FakeAsyncRedis) -> None:
    calls: list[Any] = []

    async def completion_fn(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return make_response("should not happen")

    provider = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    settings = GatewaySettings(per_request_token_ceiling=1)
    gateway = Gateway(
        settings=settings,
        registry=_registry(provider),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    with pytest.raises(BudgetExceeded):
        await gateway.complete(
            CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])
        )

    assert calls == []


class _Answer(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_structured_output_is_parsed(fake_redis: FakeAsyncRedis) -> None:
    async def completion_fn(**kwargs: Any) -> Any:
        return make_response('{"value": "42"}')

    provider = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    gateway = Gateway(
        settings=GatewaySettings(),
        registry=_registry(provider),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    result = await gateway.complete(
        CompletionRequest(
            tier=Tier.FAST,
            messages=[Message(role="user", content="hi")],
            response_model=_Answer,
        )
    )

    assert isinstance(result.parsed, _Answer)
    assert result.parsed.value == "42"


@pytest.mark.asyncio
async def test_user_uploaded_content_requires_local_tier(fake_redis: FakeAsyncRedis) -> None:
    calls: list[Any] = []

    async def completion_fn(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return make_response("nope")

    provider = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    gateway = Gateway(
        settings=GatewaySettings(),
        registry=_registry(provider),
        redis_client=fake_redis,
        completion_fn=completion_fn,
    )

    with pytest.raises(ValueError, match="Tier.LOCAL"):
        await gateway.complete(
            CompletionRequest(
                tier=Tier.REASON,
                messages=[Message(role="user", content="hi")],
                user_uploaded_content=True,
            )
        )

    assert calls == []


@pytest.mark.asyncio
async def test_cache_hit_avoids_recalling_provider(fake_redis: FakeAsyncRedis) -> None:
    calls: list[Any] = []

    async def completion_fn(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return make_response("cached answer")

    provider = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
    gateway = Gateway(
        settings=GatewaySettings(),
        registry=_registry(provider),
        redis_client=fake_redis,
        embedder=FakeEmbedder(),
        completion_fn=completion_fn,
    )
    request = CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content="hi")])

    first = await gateway.complete(request)
    second = await gateway.complete(request)

    assert len(calls) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == "cached answer"
