import pytest
from core.testing import FakeEmbedder
from fakeredis import FakeAsyncRedis
from llm.cache import ResponseCache
from llm.config import GatewaySettings
from llm.models import CompletionRequest, CompletionResult, Message, Tier, Usage


def _request(text: str = "what is the capital of France?") -> CompletionRequest:
    return CompletionRequest(tier=Tier.FAST, messages=[Message(role="user", content=text)])


def _result(text: str = "Paris") -> CompletionResult:
    return CompletionResult(
        text=text,
        parsed=None,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        provider="groq",
        model="groq/llama-3.1-8b-instant",
        tier=Tier.FAST,
        cache_hit=False,
        cache_layer=None,
        estimated_cost_usd=0.0,
        latency_ms=12.0,
        retry_count=0,
    )


@pytest.mark.asyncio
async def test_exact_match_hit_returns_without_embedder(fake_redis: FakeAsyncRedis) -> None:
    cache = ResponseCache(fake_redis, GatewaySettings(), embedder=None)
    request = _request()
    await cache.put(request, _result())

    hit = await cache.get(request)

    assert hit is not None
    assert hit.cache_hit is True
    assert hit.cache_layer == "exact"
    assert hit.text == "Paris"


@pytest.mark.asyncio
async def test_miss_without_embedder_never_touches_semantic_layer(
    fake_redis: FakeAsyncRedis,
) -> None:
    cache = ResponseCache(fake_redis, GatewaySettings(), embedder=None)

    hit = await cache.get(_request())

    assert hit is None
    assert await fake_redis.lrange("llm:cache:semantic:fast", 0, -1) == []


@pytest.mark.asyncio
async def test_semantic_hit_resolves_via_exact_layer_pointer(fake_redis: FakeAsyncRedis) -> None:
    embedder = FakeEmbedder()
    # FakeEmbedder is deterministic-but-not-semantic (hash-based, no real similarity
    # signal); a threshold of 2.0 (the max possible cosine distance) makes this test
    # about the cache's retrieval mechanics, not embedding quality.
    settings = GatewaySettings(cache_semantic_distance_threshold=2.0)
    cache = ResponseCache(fake_redis, settings, embedder=embedder)

    original = _request("what is the capital of France?")
    await cache.put(original, _result("Paris"))

    near_duplicate = _request("what's the capital city of France")
    hit = await cache.get(near_duplicate)

    assert hit is not None
    assert hit.cache_layer == "semantic"
    assert hit.text == "Paris"


@pytest.mark.asyncio
async def test_semantic_miss_beyond_distance_threshold(fake_redis: FakeAsyncRedis) -> None:
    embedder = FakeEmbedder()
    # -1.0 is below the minimum possible cosine distance (0.0) — guarantees a miss
    # regardless of the fake embedder's (non-semantic) vectors.
    settings = GatewaySettings(cache_semantic_distance_threshold=-1.0)
    cache = ResponseCache(fake_redis, settings, embedder=embedder)

    await cache.put(_request("what is the capital of France?"), _result("Paris"))

    hit = await cache.get(_request("completely unrelated question about spacecraft engines"))

    assert hit is None
