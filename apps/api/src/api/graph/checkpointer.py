from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# Neon's free tier suspends an idle compute after ~5 minutes and severs its TCP
# connections. Recycle well before that so the pool retires a connection on its own
# schedule rather than handing out one Neon has already killed.
_MAX_IDLE_SECONDS = 120.0
_MAX_LIFETIME_SECONDS = 900.0


@asynccontextmanager
async def postgres_checkpointer(database_url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Postgres is the only checkpointer outside unit tests, per CLAUDE.md — this is
    what lets a thread survive a restart on free-tier, restart-prone hosting.

    Backed by a *pool*, deliberately not AsyncPostgresSaver.from_conn_string. That
    helper opens one AsyncConnection and holds it for the lifetime of the context
    manager — i.e. the whole app process, since this is entered from the lifespan.
    It has no reconnect path, so the first time Neon suspended its compute and dropped
    that socket, every subsequent query failed on a dead connection *permanently*,
    until someone redeployed. /health kept returning ok because it never touches
    Postgres; the graph died before its first node on the initial checkpoint write.
    A pool reconnects, so a suspended Neon costs one cold start instead of an outage.

    kwargs mirror exactly what from_conn_string sets on its single connection
    (autocommit, prepare_threshold=0, dict_row) — the saver's SQL is written against
    dict rows, and its statements must not be server-prepared, so a pool that omits
    these breaks every checkpoint read.
    """
    async with AsyncConnectionPool(
        database_url,
        min_size=0,  # hold nothing open while idle — see _MAX_IDLE_SECONDS
        max_size=4,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        # Liveness check on checkout: a connection Neon killed is discarded and
        # replaced transparently instead of surfacing as a request failure.
        check=AsyncConnectionPool.check_connection,
        max_idle=_MAX_IDLE_SECONDS,
        max_lifetime=_MAX_LIFETIME_SECONDS,
    ) as pool:
        await pool.open(wait=True, timeout=30)
        checkpointer = AsyncPostgresSaver(conn=pool)
        await checkpointer.setup()
        yield checkpointer
