"""Schema migrations via Alembic — see docs/adr and the discussion that replaced the
original hand-rolled runner: yoyo-migrations (the initially-considered lightweight
option) only supports Postgres via psycopg2, a real blocker given every other
Postgres access point in this project uses psycopg3. Alembic has no such blocker —
it runs on a plain SQLAlchemy engine, and SQLAlchemy has supported the psycopg3
dialect (`postgresql+psycopg://`) since 2.0 — so per CLAUDE.md's "use the library"
rule, there was no concrete reason left to hand-roll this.

Migrations are raw SQL via `op.execute()` (see migrations/versions/) — no ORM models,
consistent with this project's no-ORM convention.
"""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def apply_migrations_sync(database_url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")


async def apply_migrations(database_url: str) -> None:
    """Runs Alembic's (synchronous) migration runner in a thread. Alembic itself
    recommends a sync engine for migrations even in async applications — this is a
    one-off startup/deploy operation, not part of the async hot path, and forcing
    Alembic into async mode has known event-loop complications."""
    await asyncio.to_thread(apply_migrations_sync, database_url)
