import pytest
from fakeredis import FakeAsyncRedis
from llm.budget import BudgetGuard, _day_key
from llm.config import GatewaySettings
from llm.errors import BudgetExceeded


@pytest.mark.asyncio
async def test_reserve_raises_per_request_without_touching_redis(
    fake_redis: FakeAsyncRedis,
) -> None:
    settings = GatewaySettings(per_request_token_ceiling=100, per_day_token_ceiling=100_000)
    guard = BudgetGuard(fake_redis, settings)

    with pytest.raises(BudgetExceeded) as exc_info:
        await guard.reserve(101)

    assert exc_info.value.scope == "per_request"
    assert await fake_redis.get(_day_key()) is None


@pytest.mark.asyncio
async def test_reserve_raises_per_day_and_rolls_back(fake_redis: FakeAsyncRedis) -> None:
    settings = GatewaySettings(per_request_token_ceiling=1000, per_day_token_ceiling=150)
    guard = BudgetGuard(fake_redis, settings)

    await guard.reserve(100)
    with pytest.raises(BudgetExceeded) as exc_info:
        await guard.reserve(100)

    assert exc_info.value.scope == "per_day"
    assert int(await fake_redis.get(_day_key())) == 100


@pytest.mark.asyncio
async def test_reconcile_adjusts_by_delta_including_refund(fake_redis: FakeAsyncRedis) -> None:
    settings = GatewaySettings(per_request_token_ceiling=1000, per_day_token_ceiling=100_000)
    guard = BudgetGuard(fake_redis, settings)

    reservation = await guard.reserve(100)
    await guard.reconcile(reservation, actual_tokens=60)

    assert int(await fake_redis.get(_day_key())) == 60


@pytest.mark.asyncio
async def test_release_fully_refunds(fake_redis: FakeAsyncRedis) -> None:
    settings = GatewaySettings(per_request_token_ceiling=1000, per_day_token_ceiling=100_000)
    guard = BudgetGuard(fake_redis, settings)

    reservation = await guard.reserve(100)
    await guard.release(reservation)

    assert int(await fake_redis.get(_day_key())) == 0
