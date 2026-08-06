"""The single entry point to any model provider.

No other module in this codebase may call a provider SDK (LiteLLM or otherwise)
directly — every model call goes through `Gateway.complete()`. See
`.claude/rules/llm-gateway.md` for the full contract this module implements: tiered
routing with per-tier fallback chains (`llm.registry`), retry with backoff respecting
`Retry-After`, a per-provider concurrency semaphore, a pre-spend budget guard, a
Redis-backed cache (exact-match in front of a brute-force semantic layer — see
`llm.cache` for why it isn't `redisvl`), Pydantic v2 structured output, and an OTel
span per call.

`completion_fn` (constructor kwarg, defaults to `litellm.acompletion`) is the
testability seam. Whatever is passed in must return an object shaped like LiteLLM's
`ModelResponse`: `.choices[0].message.content: str` and
`.usage.{prompt_tokens,completion_tokens,total_tokens}: int`. Unit tests inject a fake
with that shape; nothing in this module makes a real network call unless the default
`litellm.acompletion` is used.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import litellm
import structlog
from core.ports import Embedder
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, ValidationError
from redis.asyncio import Redis
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm.budget import BudgetGuard, Reservation
from llm.cache import ResponseCache
from llm.config import GatewaySettings
from llm.errors import AllProvidersExhausted, BudgetExceeded
from llm.models import CompletionRequest, CompletionResult, Tier, Usage
from llm.registry import ProviderModel, TierRegistry, build_default_registry
from telemetry import get_logger

RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
    BadGatewayError,
    APIConnectionError,
    Timeout,
    ValidationError,
)


def _extract_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Whether the outer chain-fallback loop and the per-provider tenacity retry
    should treat `exc` as transient. LiteLLM's exception *class* is not fully
    reliable for this on its own — a live run against Gemini's free tier surfaced a
    quota-exceeded 429 mapped to `BadRequestError`, a type this gateway doesn't
    otherwise treat as retryable (it's normally a caller-bug signal, not a capacity
    one). `BadRequestError` still carries the correct status code on `.status_code`/
    `.response.status_code` in that case, so checking the real status code in
    addition to the known-retryable exception types catches it. The type check alone
    under-catches (as observed); the status-code check alone would also treat a
    genuinely malformed request as transient if it weren't gated by the type check
    first, so both are checked."""
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    status_code = _extract_status_code(exc)
    return status_code is not None and (status_code == 429 or status_code >= 500)


def _extract_retry_after(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _make_wait(settings: GatewaySettings) -> Callable[[RetryCallState], float]:
    base_wait = wait_exponential_jitter(
        initial=settings.retry_backoff_initial_seconds,
        max=settings.retry_backoff_max_seconds,
    )

    def _wait(retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc is not None:
            retry_after = _extract_retry_after(exc)
            if retry_after is not None:
                return retry_after
        return base_wait(retry_state)

    return _wait


class Gateway:
    def __init__(
        self,
        *,
        settings: GatewaySettings | None = None,
        registry: TierRegistry | None = None,
        redis_client: Redis | None = None,
        embedder: Embedder | None = None,
        completion_fn: Callable[..., Awaitable[Any]] = litellm.acompletion,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._settings = settings or GatewaySettings()
        self._registry = registry or build_default_registry(self._settings)
        self._redis = redis_client or Redis.from_url(self._settings.redis_url)
        self._cache = ResponseCache(self._redis, self._settings, embedder)
        self._budget = BudgetGuard(self._redis, self._settings)
        self._completion_fn = completion_fn
        self._logger = logger or get_logger(__name__)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._tracer = trace.get_tracer(__name__)

    def _semaphore_for(self, provider: ProviderModel) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(provider.provider)
        if semaphore is None:
            semaphore = asyncio.Semaphore(provider.max_concurrency)
            self._semaphores[provider.provider] = semaphore
        return semaphore

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        start = time.monotonic()
        with self._tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("llm.tier", request.tier.value)
            try:
                return await self._complete(request, span, start)
            except (BudgetExceeded, AllProvidersExhausted) as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    async def _complete(
        self, request: CompletionRequest, span: Span, start: float
    ) -> CompletionResult:
        if request.user_uploaded_content and request.tier is not Tier.LOCAL:
            raise ValueError(
                "user-uploaded content must be routed to Tier.LOCAL, or refused — "
                "see .claude/rules/llm-gateway.md"
            )

        cached = await self._cache.get(request)
        if cached is not None:
            span.set_attribute("llm.cache_hit", True)
            span.set_attribute("llm.cache_layer", cached.cache_layer or "")
            elapsed_ms = (time.monotonic() - start) * 1000
            return cached.model_copy(update={"latency_ms": elapsed_ms})
        span.set_attribute("llm.cache_hit", False)

        chain = self._registry.fallback_chain(request.tier)
        estimated_tokens = self._estimate_tokens(request, chain[0])
        reservation = await self._budget.reserve(estimated_tokens)

        total_attempts = 0
        last_exc: BaseException | None = None
        for provider in chain:
            try:
                response, parsed, attempts = await self._call_provider(provider, request)
            except Exception as exc:
                if not _is_retryable(exc):
                    await self._budget.release(reservation)
                    raise
                total_attempts += self._settings.same_provider_retry_attempts
                last_exc = exc
                self._logger.warning(
                    "llm.provider_failed",
                    provider=provider.provider,
                    model=provider.model,
                    tier=request.tier.value,
                    error=type(exc).__name__,
                    status_code=_extract_status_code(exc),
                )
                continue
            total_attempts += attempts
            return await self._finish(
                request, provider, response, parsed, reservation, total_attempts - 1, span, start
            )

        await self._budget.release(reservation)
        raise AllProvidersExhausted(request.tier) from last_exc

    def _estimate_tokens(self, request: CompletionRequest, provider: ProviderModel) -> int:
        prompt_tokens = litellm.token_counter(
            model=provider.model,
            messages=[m.model_dump() for m in request.messages],
        )
        return prompt_tokens + request.max_tokens

    async def _call_provider(
        self, provider: ProviderModel, request: CompletionRequest
    ) -> tuple[Any, BaseModel | None, int]:
        attempts = 0
        parsed: BaseModel | None = None
        response: Any = None
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._settings.same_provider_retry_attempts),
            wait=_make_wait(self._settings),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        async for attempt in retryer:
            attempts += 1
            with attempt:
                async with self._semaphore_for(provider):
                    response = await self._completion_fn(
                        model=provider.model,
                        messages=[m.model_dump() for m in request.messages],
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        timeout=self._settings.request_timeout_seconds,
                        response_format=request.response_model,
                    )
                content = response.choices[0].message.content
                if request.response_model is not None:
                    parsed = request.response_model.model_validate_json(content)
        return response, parsed, attempts

    async def _finish(
        self,
        request: CompletionRequest,
        provider: ProviderModel,
        response: Any,
        parsed: BaseModel | None,
        reservation: Reservation,
        retry_count: int,
        span: Span,
        start: float,
    ) -> CompletionResult:
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        await self._budget.reconcile(reservation, usage.total_tokens)

        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=provider.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
            estimated_cost = prompt_cost + completion_cost
        except Exception:
            estimated_cost = 0.0

        result = CompletionResult(
            text=response.choices[0].message.content,
            parsed=parsed,
            usage=usage,
            provider=provider.provider,
            model=provider.model,
            tier=request.tier,
            cache_hit=False,
            cache_layer=None,
            estimated_cost_usd=estimated_cost,
            latency_ms=(time.monotonic() - start) * 1000,
            retry_count=retry_count,
        )
        await self._cache.put(request, result)

        span.set_attribute("llm.provider", provider.provider)
        span.set_attribute("llm.model", provider.model)
        span.set_attribute("llm.tokens_in", usage.prompt_tokens)
        span.set_attribute("llm.tokens_out", usage.completion_tokens)
        span.set_attribute("llm.estimated_cost_usd", estimated_cost)
        span.set_attribute("llm.retry_count", retry_count)

        return result
