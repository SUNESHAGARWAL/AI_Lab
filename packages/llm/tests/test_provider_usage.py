import pytest
from fakeredis import FakeAsyncRedis
from llm.provider_usage import ProviderUsageTracker


@pytest.mark.asyncio
async def test_count_is_zero_when_unset(fake_redis: FakeAsyncRedis) -> None:
    tracker = ProviderUsageTracker(fake_redis)
    assert await tracker.count("gemini") == 0


@pytest.mark.asyncio
async def test_increment_and_count_round_trip(fake_redis: FakeAsyncRedis) -> None:
    tracker = ProviderUsageTracker(fake_redis)

    for _ in range(3):
        await tracker.increment("gemini")

    assert await tracker.count("gemini") == 3
    assert await tracker.count("groq") == 0  # independent per-provider


@pytest.mark.asyncio
async def test_increment_sets_ttl_bounded_by_a_day(fake_redis: FakeAsyncRedis) -> None:
    tracker = ProviderUsageTracker(fake_redis)
    await tracker.increment("gemini")

    ttl = await fake_redis.ttl(tracker._key("gemini"))
    assert 0 < ttl <= 24 * 60 * 60


@pytest.mark.asyncio
async def test_is_near_ceiling_boundary(fake_redis: FakeAsyncRedis) -> None:
    tracker = ProviderUsageTracker(fake_redis)

    for _ in range(9):
        await tracker.increment("gemini")
    assert await tracker.is_near_ceiling("gemini", ceiling=10, soft_fraction=0.9) is True

    await tracker.increment("groq")  # unrelated provider, shouldn't affect gemini
    assert await tracker.count("gemini") == 9


@pytest.mark.asyncio
async def test_is_near_ceiling_false_below_threshold(fake_redis: FakeAsyncRedis) -> None:
    tracker = ProviderUsageTracker(fake_redis)

    for _ in range(8):
        await tracker.increment("gemini")

    assert await tracker.is_near_ceiling("gemini", ceiling=10, soft_fraction=0.9) is False
