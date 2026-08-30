"""Per-IP rate limiting for the public demo's LIVE query path only — cached example
replays (apps/web/lib/replay-client.ts) never reach this endpoint, so they're
unaffected by design, not by a special-case check here."""

from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis

from api.config import Settings
from telemetry import get_logger

# Give an allowance back only if the key is still there. A bare DECR on an expired or
# missing key *creates* it at -1 with no TTL, which both leaks the key forever and
# hands the next hour's visitor a free extra query. Clamping at zero keeps a double
# refund (or a refund racing the hour rollover) from doing the same thing.
_REFUND_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  return -1
end
local remaining = redis.call('DECR', KEYS[1])
if remaining < 0 then
  redis.call('INCR', KEYS[1])
  return 0
end
return remaining
"""


logger = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int
    # The exact bucket this call consumed. Refunds target *this* key rather than
    # recomputing it later: a request that starts at 10:59:59 and fails at 11:00:01
    # would otherwise credit the next hour's bucket, granting a free query.
    key: str = ""


def _hour_key(client_key: str) -> str:
    return f"api:ratelimit:{client_key}:{datetime.now(UTC).strftime('%Y%m%d%H')}"


class RateLimiter:
    """Fixed-window per-client limiter, one INCR+EXPIRE on an hour-bucketed key — same
    shape as llm.budget.BudgetGuard's day-bucketed key, deliberately not a
    sliding-window algorithm (unnecessary precision for "N per hour" on a portfolio
    demo).

    An exceeded limit is not an error — `check` reports it and the caller decides what
    it means. An *unreachable Redis* is a different matter and does propagate: failing
    open would drop the demo's only per-visitor spend control at exactly the moment
    llm.budget.BudgetGuard's day ceiling is also blind (it shares this Redis), so a
    Redis outage must stop live queries rather than uncap them. The route turns that
    into a terminal error frame; see api.routes.stream._guarded_sse_body.
    """

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings

    async def check(self, client_key: str) -> RateLimitResult:
        limit = self._settings.live_query_rate_limit_per_hour
        key = _hour_key(client_key)
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, 3600)
        return RateLimitResult(
            allowed=count <= limit, remaining=max(0, limit - count), limit=limit, key=key
        )

    async def refund(self, key: str) -> None:
        """Return an allowance consumed by a run that failed through no fault of the
        visitor. Nobody should lose one of their few live queries to our dead database
        connection — that is what made the outage doubly punishing: a visitor could
        burn every attempt they had on requests that never produced an answer.

        Deliberately narrow. The caller refunds only our-fault failures, never a
        completed answer, an abstention, or a spent cost ceiling — see
        api.routes.stream._REFUNDABLE_ERROR_REASONS for the reasoning on each.

        Best-effort by design: a refund that fails must never turn a handled error into
        an unhandled one, and the bucket expires within the hour regardless.
        """
        if not key:
            return
        try:
            await self._redis.eval(_REFUND_SCRIPT, 1, key)
        except Exception:
            # Swallowed, never re-raised: see docstring. But logged rather than
            # silently passed — a refund that never actually works (an EVAL the
            # provider rejects, say) would otherwise be invisible forever, quietly
            # charging visitors for our own failures.
            logger.warning("ratelimit.refund_failed", key=key, exc_info=True)
