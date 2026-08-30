"""Liveness (/health) and readiness (/ready) probes — deliberately two endpoints, not
one. See the note on `ready` for why /health must stay dependency-free."""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from api.config import Settings, get_settings
from telemetry import get_logger

router = APIRouter()
logger = get_logger(__name__)

# A probe that can hang is not a probe. Short enough that a wedged dependency reports
# as down rather than holding the request open; long enough to absorb a Neon cold start
# on the free tier, which routinely takes a few seconds after a suspend.
_PROBE_TIMEOUT_SECONDS = 8.0


@router.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
    """Liveness only: is the process up and serving? Never touches a dependency.

    Railway probes this with restartPolicyType ON_FAILURE, so making it dependency-aware
    would turn a routine Neon suspend into a restart loop — the container would be
    killed for a database that is merely asleep. Use /ready to ask whether the app can
    actually serve a query.
    """
    return {"status": "ok", "env": settings.app_env}


async def _check(name: str, probe: Any) -> tuple[str, str | None]:
    try:
        await asyncio.wait_for(probe, timeout=_PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        return name, f"timed out after {_PROBE_TIMEOUT_SECONDS:g}s"
    except Exception as exc:
        return name, type(exc).__name__
    return name, None


async def _postgres_ok(pool: Any) -> None:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1")


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: can this instance actually answer a query right now?

    This exists because of a real outage. The checkpointer held a single Postgres
    connection that Neon severed on suspend, every query failed from that moment on,
    and /health went on returning ok the whole time because it never touched the
    database — so from the outside a totally broken backend was indistinguishable from
    a healthy one, and it looked like a frontend bug. Anything the request path depends
    on gets checked here, and the failing dependency is named rather than hidden behind
    a bare 503.
    """
    state = request.app.state
    results = dict(
        await asyncio.gather(
            _check("postgres", _postgres_ok(state.pool)),
            _check("redis", state.redis_client.ping()),
        )
    )
    failures = {name: err for name, err in results.items() if err is not None}
    if failures:
        # The one place a dependency failure is worth an error log: it means this
        # instance is serving nothing, and it is the signal that was missing before.
        logger.error("api.not_ready", failures=failures)
    return JSONResponse(
        status_code=503 if failures else 200,
        content={
            "status": "degraded" if failures else "ok",
            "checks": {name: err or "ok" for name, err in results.items()},
        },
    )
