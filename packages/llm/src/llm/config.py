from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

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
