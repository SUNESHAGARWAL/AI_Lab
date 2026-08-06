import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No ORM models in this project (raw SQL migrations only, per CLAUDE.md's no-ORM
# convention) — nothing to autogenerate against.
target_metadata = None


def _database_url() -> str:
    # Config.set_main_option("sqlalchemy.url", ...) (used by retrieval.migrate for
    # programmatic invocation) takes precedence; DATABASE_URL env var is the fallback
    # for direct `alembic upgrade head` CLI usage.
    url = config.get_main_option("sqlalchemy.url") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL must be set (or sqlalchemy.url configured) to run migrations"
        )
    # SQLAlchemy's psycopg3 dialect needs an explicit +psycopg driver name — every
    # other Postgres access point in this project already uses psycopg3.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
