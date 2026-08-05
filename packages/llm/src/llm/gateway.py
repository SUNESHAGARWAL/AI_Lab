"""The single entry point to any model provider.

No other module in this codebase may call a provider SDK (LiteLLM or otherwise)
directly — every model call goes through this gateway. Once implemented, this module
owns: provider routing and free-tier fallback chains, retry with backoff (LiteLLM's
token_bucket policy, not naive exponential backoff), response caching, budget
enforcement, and OpenTelemetry spans + token-count/hash-only logging.

Routing/retry/caching/budget logic is not implemented yet — this is the workspace
scaffold stage (see docs/adr/0001-orchestration.md).
"""
