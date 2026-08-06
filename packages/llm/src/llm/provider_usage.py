"""Local, self-reported daily request counters, per provider.

Distinct from `llm.budget.BudgetGuard`: that enforces *our own* token spend ceiling
across every provider. This tracks how close we are to *one specific provider's own*
published rate limit, purely to decide whether attempting a call is worth risking a
429 — soft and best-effort, since it only knows about calls this gateway itself made
(not a provider's true server-side count, e.g. from another process or environment).
"""

from datetime import UTC, datetime, timedelta

from redis.asyncio import Redis

_KEY_PREFIX = "llm:provider_usage"


def _utc_date_str() -> str:
    return datetime.now(UTC).date().isoformat()


def _seconds_until_next_utc_midnight() -> int:
    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds())


class ProviderUsageTracker:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def _key(self, provider: str) -> str:
        return f"{_KEY_PREFIX}:{provider}:{_utc_date_str()}"

    async def increment(self, provider: str) -> None:
        key = self._key(provider)
        await self._redis.incr(key)
        # Safe to call every time: recomputing "seconds until midnight" still
        # converges to the same absolute expiry instant within a given UTC day.
        await self._redis.expire(key, _seconds_until_next_utc_midnight())

    async def count(self, provider: str) -> int:
        value = await self._redis.get(self._key(provider))
        return int(value) if value is not None else 0

    async def is_near_ceiling(self, provider: str, ceiling: int, soft_fraction: float) -> bool:
        return await self.count(provider) >= ceiling * soft_fraction
