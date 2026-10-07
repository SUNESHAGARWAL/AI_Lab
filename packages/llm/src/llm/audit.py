"""Attempt recorder: maps one provider call attempt to one content-free GenAI inference record.

The GenAI handler from `opentelemetry-util-genai` (built by
`telemetry.audit_trace.open_audit_sink`) owns the span, its timing and the `gen_ai.*` names. This
module only fills the handler's fields from a `CompletionRequest` and a LiteLLM-shaped response, and
adds the `llm_audit.*` attributes the `llm-audit` OTel adapter maps. No message, system or
completion text and no provider error text is ever written: the request is reduced to sha256 hashes
and a token count, and a failure keeps only the exception's short type name and HTTP status.

Recording must never change gateway behaviour, so every public method catches its own errors and
logs `llm.audit_trace_failed` with the error type only.
"""

import hashlib
import itertools
import json
from collections.abc import Callable
from typing import Protocol

import litellm
import structlog
from opentelemetry.trace import format_span_id, format_trace_id
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import Error

from llm.models import CompletionRequest, Message
from llm.registry import ProviderModel
from telemetry import get_logger

# Attribute names of the attempt-record contract that the library does not set itself.
ATTEMPT = "llm_audit.attempt"
RETRY_OF = "llm_audit.retry_of"
FEATURE = "llm_audit.feature"
ENV = "llm_audit.env"
PROMPT_HASH = "llm_audit.prompt_hash"
PREFIX_HASH = "llm_audit.prefix_hash"
PREFIX_TOKENS = "llm_audit.prefix_tokens"
HTTP_STATUS = "http.response.status_code"
# Set by the library from the invocation fields, but only when non-zero (`value or None` in
# `InferenceInvocation._get_attributes`). A reported 0 is a measurement, not a missing field, and
# the audit tells "no cache hits" from "cache tokens not reported" by it, so a reported 0 is
# written explicitly under the same name.
CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"

FAILED_EVENT = "llm.audit_trace_failed"

StatusCodeOf = Callable[[BaseException], int | None]


class AttemptHandle(Protocol):
    # "0x<trace_id>:0x<span_id>", the call id the llm-audit adapter builds from the line's context.
    # None only when the record could not be started (logged); pass it on as `retry_of` regardless.
    call_id: str | None

    def set_response(self, response: object) -> None: ...
    def succeed(self) -> None: ...
    def fail(self, exc: BaseException) -> None: ...


def _short_type(exc: BaseException) -> str:
    return type(exc).__name__


def _sha256(messages: list[Message]) -> str:
    payload = json.dumps([m.model_dump() for m in messages], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _count(value: object) -> int | None:
    # Token counts only; anything else (None, a mock, a bool) stays absent, never guessed (1.5).
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class _Attempt:
    def __init__(
        self,
        invocation: InferenceInvocation,
        status_code: StatusCodeOf,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._invocation = invocation
        self._status_code = status_code
        self._logger = logger
        ctx = invocation.span.get_span_context()
        self.call_id: str | None = (
            f"0x{format_trace_id(ctx.trace_id)}:0x{format_span_id(ctx.span_id)}"
        )

    def set_response(self, response: object) -> None:
        try:
            self._set_response(response)
        except Exception as exc:
            self._logger.warning(FAILED_EVENT, step="set_response", error_type=_short_type(exc))

    def _set_response(self, response: object) -> None:
        inv = self._invocation
        model = getattr(response, "model", None)
        if isinstance(model, str) and model:
            inv.response_model_name = model
        choices = getattr(response, "choices", None)
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
            if isinstance(finish_reason, str) and finish_reason:
                inv.finish_reasons = [finish_reason]
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        inv.input_tokens = _count(getattr(usage, "prompt_tokens", None))
        inv.output_tokens = _count(getattr(usage, "completion_tokens", None))
        cached = _count(
            getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        )
        reasoning = _count(
            getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        )
        inv.cache_read_input_tokens = cached
        inv.thinking_tokens = reasoning
        for name, value in ((CACHE_READ_TOKENS, cached), (REASONING_TOKENS, reasoning)):
            if value == 0:
                inv.attributes[name] = 0

    def succeed(self) -> None:
        try:
            self._invocation.stop()
        except Exception as exc:
            self._logger.warning(FAILED_EVENT, step="succeed", error_type=_short_type(exc))

    def fail(self, exc: BaseException) -> None:
        name = _short_type(exc)
        try:
            status = self._status_code(exc)
            if status is not None:
                self._invocation.attributes[HTTP_STATUS] = status
        except Exception as lookup_exc:
            self._logger.warning(
                FAILED_EVENT, step="status_code", error_type=_short_type(lookup_exc)
            )
        try:
            # An explicit Error, not the exception: `Error.from_exception` would use str(exc), the
            # provider's error text, as the span status description.
            self._invocation.fail(Error(message=name, type=name))
        except Exception as fail_exc:
            self._logger.warning(FAILED_EVENT, step="fail", error_type=_short_type(fail_exc))


class _NoRecord:
    """Returned when a record could not be started; every method is a no-op."""

    call_id: str | None = None

    def set_response(self, response: object) -> None:
        return None

    def succeed(self) -> None:
        return None

    def fail(self, exc: BaseException) -> None:
        return None


class AttemptRecorder:
    """Starts one inference record per provider call attempt on the audit sink's handler.

    `status_code` reads the HTTP status from a provider exception; the gateway passes its own
    `_extract_status_code`, so this module needs no import of the gateway.
    """

    def __init__(
        self,
        handler: TelemetryHandler,
        *,
        env: str,
        status_code: StatusCodeOf,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._handler = handler
        self._env = env
        self._status_code = status_code
        self._logger = logger or get_logger(__name__)

    def start(
        self,
        *,
        provider: ProviderModel,
        request: CompletionRequest,
        attempt: int,
        retry_of: str | None,
    ) -> AttemptHandle:
        try:
            attributes = self._attributes(provider, request, attempt, retry_of)
            invocation = self._handler.inference(
                provider.provider,
                request_model=provider.model,
                operation_name="chat",
                error_type_resolver=_short_type,
            )
        except Exception as exc:
            self._logger.warning(FAILED_EVENT, step="start", error_type=_short_type(exc))
            return _NoRecord()
        invocation.max_tokens = request.max_tokens
        invocation.temperature = request.temperature
        invocation.attributes.update(attributes)
        return _Attempt(invocation, self._status_code, self._logger)

    def _attributes(
        self,
        provider: ProviderModel,
        request: CompletionRequest,
        attempt: int,
        retry_of: str | None,
    ) -> dict[str, str | int]:
        # Computed before the span starts, so a hashing or token-count error opens no record.
        attributes: dict[str, str | int] = {
            ATTEMPT: attempt,
            ENV: self._env,
            PROMPT_HASH: _sha256(request.messages),
        }
        if retry_of is not None:
            attributes[RETRY_OF] = retry_of
        if request.feature is not None:
            attributes[FEATURE] = request.feature
        prefix = list(itertools.takewhile(lambda m: m.role == "system", request.messages))
        if prefix:
            attributes[PREFIX_HASH] = _sha256(prefix)
            attributes[PREFIX_TOKENS] = litellm.token_counter(
                model=provider.model, messages=[m.model_dump() for m in prefix]
            )
        return attributes
