from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Shared with apps/api/src/api/config.py's Settings.app_env — same unprefixed
    # APP_ENV var, so one .env entry drives both, not a separate LLM_APP_ENV.
    app_env: str = Field(default="development", validation_alias="APP_ENV")

    request_timeout_seconds: float = 30.0
    same_provider_retry_attempts: int = 2
    retry_backoff_initial_seconds: float = 1.0
    retry_backoff_max_seconds: float = 20.0

    redis_url: str = "redis://localhost:6379/0"
    cache_exact_ttl_seconds: int = 3600
    cache_semantic_ttl_seconds: int = 3600
    cache_semantic_max_candidates: int = 200
    cache_semantic_distance_threshold: float = 0.05

    per_request_token_ceiling: int = 4000
    per_day_token_ceiling: int = 200_000

    provider_concurrency_overrides: dict[str, int] = {}
    provider_rpm_overrides: dict[str, int] = {}

    # Gemini's published free-tier RPD as of this writing (gemini-2.0-flash).
    # Re-verify periodically against https://ai.google.dev/gemini-api/docs/rate-limits
    # before trusting this — Google cut free-tier quotas 50-80% in December 2025, and
    # .claude/rules/llm-gateway.md already warns these shift roughly monthly without
    # notice. This is a soft, self-imposed limit based on our own local tracked usage
    # (see llm.provider_usage), not something Gemini reports back to us.
    gemini_daily_request_ceiling: int = 1500
    gemini_soft_limit_fraction: float = 0.9
