from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)


def create_pool(database_url: str) -> AsyncConnectionPool:
    """Constructs (but does not open) a pool whose connections have the pgvector type
    adapter registered — registration is per-connection, not global, so it has to run
    via the pool's configure callback on every new connection, not once at import
    time."""
    return AsyncConnectionPool(database_url, configure=_configure, open=False)
