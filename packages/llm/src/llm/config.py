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

    # 8000 was too tight once the generator's max_tokens was raised to 4096 (see
    # api.graph.nodes.make_generator_node's comment: deepseek/deepseek-reasoner, the
    # reason tier's primary model, spends real completion tokens on an internal
    # reasoning pass before visible content, verified empirically). Worst case:
    # DEFAULT_RERANK_TOP_N (5) full AI Act/GDPR article chunks (~5000 tokens) + the
    # question + 4096 max_tokens can approach 9500-10000 before this ceiling is
    # even relevant — 16000 leaves real headroom, not just enough to clear today's
    # specific number.
    per_request_token_ceiling: int = 16000
    # 200_000 was sized for Groq-free-tier-style caution and tripped mid-way
    # through a real 36-item Layer 3 run within one debugging day. Per
    # docs/adr/0005-deepseek-primary-groq-free-fallback.md's cost model, DeepSeek
    # (the primary provider) prices this project's whole estimated *monthly* usage
    # (1.8M-10.8M tokens) at $0.32-1.89 total — a 2,000,000/day self-imposed cap
    # costs well under $1/day even fully spent, while still being a real ceiling,
    # not effectively unlimited.
    per_day_token_ceiling: int = 2_000_000

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
