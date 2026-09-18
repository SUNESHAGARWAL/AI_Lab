from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str
    redis_url: str
    # Migrations run over this URL instead of database_url when set. Needed against
    # Neon: database_url is the pooled (PgBouncer transaction-mode) connection that
    # production runtime traffic uses, but running Alembic's migration + the pool's own
    # startup concurrently over a pooled connection crashes uvicorn's ASGI process outright
    # (reproduced locally — silent, no traceback, only under uvicorn + pooled, not a
    # bare script and not the direct connection). Point this at Neon's direct
    # connection string in production; falls back to database_url for local dev,
    # where Postgres isn't pooled at all.
    migrations_database_url: str | None = None

    # The demo endpoint is public — a length cap is the input guard required by
    # CLAUDE.md's security rules for every public path.
    max_query_length: int = 2000
    # Live (non-cached) queries per IP per hour. Cached example replays are exempt —
    # see api.ratelimit's module docstring. Deliberately conservative for a portfolio
    # demo running on my own API keys; tune via env var, no redeploy needed.
    live_query_rate_limit_per_hour: int = 5

    # Where the rate limiter reads the visitor's address from. The socket address is the
    # platform's edge proxy, so a header is the only source — but which header is safe
    # depends entirely on what the proxy in front does with it:
    #   - trusted_proxy_hops = 0: use the header's value as-is. Only safe when the proxy
    #     *overwrites* it (e.g. X-Real-IP behind a proxy that sets it).
    #   - trusted_proxy_hops = N: the header is a comma-separated list the proxy
    #     *appends* to (X-Forwarded-For); take the Nth entry from the right. Everything
    #     further left is client-supplied and spoofable. Cloud Run: x-forwarded-for.
    # uvicorn's ProxyHeadersMiddleware can't do this — it needs the proxies' IPs, and
    # Google's front ends aren't a fixed published set.
    client_ip_header: str = "x-real-ip"
    trusted_proxy_hops: int = Field(default=0, ge=0)

    # Comma-separated origins CORSMiddleware allows — never a wildcard in production.
    # Defaults to the local frontend dev server so `just dev` needs no .env entry;
    # production (Cloud Run) MUST set this explicitly to the real Vercel domain(s).
    frontend_origin: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
