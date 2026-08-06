from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis

from llm.config import GatewaySettings
from llm.errors import BudgetExceeded


@dataclass(frozen=True)
class Reservation:
    estimated_tokens: int
    day_key: str


def _day_key() -> str:
    return f"llm:budget:tokens:{datetime.now(UTC).date().isoformat()}"


class BudgetGuard:
    """Reserve-then-reconcile token budget, enforced *before* any provider is called.
    A plain check-then-spend would race under concurrent requests; reserving the
    estimate atomically (and refunding on rollback/reconcile) does not."""

    def __init__(self, redis: Redis, settings: GatewaySettings) -> None:
        self._redis = redis
        self._settings = settings

    async def reserve(self, estimated_tokens: int) -> Reservation:
        ceiling = self._settings.per_request_token_ceiling
        if estimated_tokens > ceiling:
            raise BudgetExceeded("per_request", estimated_tokens, ceiling)

        day_key = _day_key()
        new_total = await self._redis.incrby(day_key, estimated_tokens)
        day_ceiling = self._settings.per_day_token_ceiling
        if new_total > day_ceiling:
            await self._redis.decrby(day_key, estimated_tokens)
            raise BudgetExceeded("per_day", estimated_tokens, day_ceiling)

        return Reservation(estimated_tokens=estimated_tokens, day_key=day_key)

    async def reconcile(self, reservation: Reservation, actual_tokens: int) -> None:
        delta = actual_tokens - reservation.estimated_tokens
        if delta != 0:
            await self._redis.incrby(reservation.day_key, delta)

    async def release(self, reservation: Reservation) -> None:
        await self._redis.decrby(reservation.day_key, reservation.estimated_tokens)
