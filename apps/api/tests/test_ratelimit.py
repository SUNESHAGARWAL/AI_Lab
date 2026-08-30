import pytest
from api.config import Settings
from api.ratelimit import RateLimiter
from api.routes.stream import client_key_for
from fakeredis import FakeAsyncRedis
from starlette.requests import Request


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


def _request(headers: dict[str, str], client_host: str | None) -> Request:
    """Minimal ASGI scope — enough for Starlette's Request to expose headers/client."""
    scope: dict[str, object] = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (client_host, 12345) if client_host else None,
    }
    return Request(scope)  # type: ignore[arg-type]


def test_client_key_prefers_x_real_ip_over_the_proxy_socket_address() -> None:
    # Behind Railway the socket address is the edge proxy — keying on it would put
    # every visitor in one shared rate-limit bucket.
    request = _request({"x-real-ip": "203.0.113.7"}, client_host="10.0.0.1")
    assert client_key_for(request) == "203.0.113.7"


def test_client_key_ignores_spoofable_x_forwarded_for() -> None:
    # Railway appends to XFF and keeps client-supplied entries, so it must not be
    # trusted — a caller could otherwise rotate values to skip the limit.
    request = _request(
        {"x-forwarded-for": "1.2.3.4", "x-real-ip": "203.0.113.7"}, client_host="10.0.0.1"
    )
    assert client_key_for(request) == "203.0.113.7"


def test_client_key_falls_back_to_socket_address_without_a_proxy() -> None:
    assert client_key_for(_request({}, client_host="198.51.100.9")) == "198.51.100.9"


def test_client_key_handles_a_missing_client() -> None:
    assert client_key_for(_request({}, client_host=None)) == "unknown"


@pytest.mark.asyncio
async def test_refund_returns_the_allowance_to_the_same_bucket() -> None:
    limiter = RateLimiter(FakeAsyncRedis(), _settings(limit=2))

    spent = await limiter.check("1.2.3.4")
    assert (await limiter.check("1.2.3.4")).allowed is True  # allowance now used up

    await limiter.refund(spent.key)

    # Exactly one slot comes back — not a reset of the window.
    assert (await limiter.check("1.2.3.4")).allowed is True
    assert (await limiter.check("1.2.3.4")).allowed is False


@pytest.mark.asyncio
async def test_refund_never_resurrects_an_expired_bucket() -> None:
    """A bare DECR on a missing key creates it at -1 with no TTL — a key that leaks
    forever and silently grants the next visitor in that bucket an extra query. The
    refund must be a no-op when the window has already rolled over."""
    redis = FakeAsyncRedis()
    limiter = RateLimiter(redis, _settings(limit=1))

    await limiter.refund("api:ratelimit:1.2.3.4:2020010100")

    assert [k async for k in redis.scan_iter("api:ratelimit:*")] == []
    assert (await limiter.check("1.2.3.4")).allowed is True


@pytest.mark.asyncio
async def test_repeated_refunds_cannot_drive_the_counter_negative() -> None:
    """Otherwise a single failing request refunded twice would hand out free queries
    for the rest of the hour."""
    redis = FakeAsyncRedis()
    limiter = RateLimiter(redis, _settings(limit=1))

    spent = await limiter.check("1.2.3.4")
    for _ in range(5):
        await limiter.refund(spent.key)

    assert int(await redis.get(spent.key)) == 0
    assert (await limiter.check("1.2.3.4")).allowed is True
    assert (await limiter.check("1.2.3.4")).allowed is False


@pytest.mark.asyncio
async def test_refund_is_best_effort_and_never_raises() -> None:
    """A refund runs while a failure is already being handled — it must not turn a
    handled error into an unhandled one."""

    class _BrokenRedis(FakeAsyncRedis):
        async def eval(self, *args: object, **kwargs: object) -> object:
            raise ConnectionError("Connection closed by server.")

    limiter = RateLimiter(_BrokenRedis(), _settings(limit=1))
    spent = await limiter.check("1.2.3.4")

    await limiter.refund(spent.key)  # must not raise
