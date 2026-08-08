"""Per-IP rate limiting for the public demo's LIVE query path only — cached example
replays (apps/web/lib/replay-client.ts) never reach this endpoint, so they're
unaffected by design, not by a special-case check here."""

from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis

from api.config import Settings


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int


def _hour_key(client_key: str) -> str:
    return f"api:ratelimit:{client_key}:{datetime.now(UTC).strftime('%Y%m%d%H')}"


class RateLimiter:
    """Fixed-window per-client limiter, one INCR+EXPIRE on an hour-bucketed key — same
    shape as llm.budget.BudgetGuard's day-bucketed key, deliberately not a
    sliding-window algorithm (unnecessary precision for "N per hour" on a portfolio
    demo). Never raises; the caller decides what an exceeded limit means."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings

    async def check(self, client_key: str) -> RateLimitResult:
        limit = self._settings.live_query_rate_limit_per_hour
        key = _hour_key(client_key)
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, 3600)
        return RateLimitResult(allowed=count <= limit, remaining=max(0, limit - count), limit=limit)
