from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str
    redis_url: str

    # The demo endpoint is public — a length cap is the input guard required by
    # CLAUDE.md's security rules for every public path.
    max_query_length: int = 2000
    # Live (non-cached) queries per IP per hour. Cached example replays are exempt —
    # see api.ratelimit's module docstring. Deliberately conservative for a portfolio
    # demo running on my own API keys; tune via env var, no redeploy needed.
    live_query_rate_limit_per_hour: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
