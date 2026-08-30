from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

# Neon's free tier suspends an idle compute after ~5 minutes, severing its connections.
# Retire idle connections before that window so the pool renews on its own terms.
_MAX_IDLE_SECONDS = 120.0
_MAX_LIFETIME_SECONDS = 900.0


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)


def create_pool(database_url: str) -> AsyncConnectionPool:
    """Constructs (but does not open) a pool whose connections have the pgvector type
    adapter registered — registration is per-connection, not global, so it has to run
    via the pool's configure callback on every new connection, not once at import
    time.

    `check` is what makes this survive a suspended Neon compute: psycopg_pool does not
    validate a connection on checkout by default, so after a suspend the pool cheerfully
    hands out a socket the server has already closed and the request fails. With a check
    the dead connection is discarded and replaced transparently. Same failure mode the
    checkpointer hit (api.graph.checkpointer), which had no pool to recover with at all.
    """
    return AsyncConnectionPool(
        database_url,
        configure=_configure,
        check=AsyncConnectionPool.check_connection,
        max_idle=_MAX_IDLE_SECONDS,
        max_lifetime=_MAX_LIFETIME_SECONDS,
        open=False,
    )
