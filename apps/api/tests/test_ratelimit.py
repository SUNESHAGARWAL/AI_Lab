import pytest
from api.config import Settings
from api.ratelimit import RateLimiter
from fakeredis import FakeAsyncRedis


def _settings(limit: int) -> Settings:
    return Settings(
        database_url="postgresql://x", redis_url="redis://x", live_query_rate_limit_per_hour=limit
    )


@pytest.mark.asyncio
async def test_allows_up_to_the_limit_then_blocks() -> None:
    limiter = RateLimiter(FakeAsyncRedis(), _settings(limit=3))

    results = [await limiter.check("1.2.3.4") for _ in range(4)]

    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[-1].remaining == 0
    assert results[-1].limit == 3


@pytest.mark.asyncio
async def test_different_clients_have_independent_counters() -> None:
    limiter = RateLimiter(FakeAsyncRedis(), _settings(limit=1))

    first = await limiter.check("1.1.1.1")
    second = await limiter.check("2.2.2.2")

    assert first.allowed is True
    assert second.allowed is True


@pytest.mark.asyncio
async def test_hour_bucketed_key_expires_so_the_window_rolls_over() -> None:
    redis = FakeAsyncRedis()
    limiter = RateLimiter(redis, _settings(limit=1))

    await limiter.check("1.2.3.4")

    keys = [k async for k in redis.scan_iter("api:ratelimit:*")]
    assert len(keys) == 1
    ttl = await redis.ttl(keys[0])
    assert 0 < ttl <= 3600
