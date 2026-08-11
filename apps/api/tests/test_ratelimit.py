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
