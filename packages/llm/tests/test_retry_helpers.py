from types import SimpleNamespace

import httpx
from litellm.exceptions import BadRequestError, RateLimitError
from llm.config import GatewaySettings
from llm.gateway import _extract_retry_after, _is_retryable, _make_wait


class _ExcWithHeaders(Exception):
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _ExcWithResponse(Exception):
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response


def test_extract_retry_after_from_headers_attribute() -> None:
    assert _extract_retry_after(_ExcWithHeaders({"retry-after": "3.5"})) == 3.5


def test_extract_retry_after_from_response_headers() -> None:
    exc = _ExcWithResponse(SimpleNamespace(headers={"retry-after": "2"}))
    assert _extract_retry_after(exc) == 2.0


def test_extract_retry_after_none_when_absent() -> None:
    assert _extract_retry_after(Exception("boom")) is None


def test_make_wait_prefers_retry_after_over_backoff() -> None:
    settings = GatewaySettings(retry_backoff_initial_seconds=10.0, retry_backoff_max_seconds=60.0)
    wait_fn = _make_wait(settings)

    exc = _ExcWithHeaders({"retry-after": "0.01"})
    retry_state = SimpleNamespace(
        outcome=SimpleNamespace(exception=lambda: exc), attempt_number=1
    )

    assert wait_fn(retry_state) == 0.01


def test_make_wait_falls_back_to_exponential_jitter_without_retry_after() -> None:
    settings = GatewaySettings(retry_backoff_initial_seconds=1.0, retry_backoff_max_seconds=5.0)
    wait_fn = _make_wait(settings)

    exc = Exception("boom")
    retry_state = SimpleNamespace(
        outcome=SimpleNamespace(exception=lambda: exc), attempt_number=1
    )

    wait_seconds = wait_fn(retry_state)
    assert 0 <= wait_seconds <= 5.0


def test_is_retryable_true_for_known_type() -> None:
    exc = RateLimitError(message="rate limited", llm_provider="test", model="m")
    assert _is_retryable(exc) is True


def test_is_retryable_false_for_unrelated_exception() -> None:
    assert _is_retryable(ValueError("bad input")) is False


def test_is_retryable_true_for_wrong_type_but_real_429_status() -> None:
    """Regression: a live run against Gemini's free tier surfaced a quota-exceeded
    429 mapped by LiteLLM to `BadRequestError` — a type this gateway doesn't
    otherwise treat as retryable (normally a caller-bug signal). `_is_retryable`
    must catch this via the real status code, not just the exception type."""
    response = httpx.Response(
        status_code=429, request=httpx.Request("GET", "https://example.com")
    )
    exc = BadRequestError(
        message="quota exceeded", model="m", llm_provider="test", response=response
    )
    assert exc.status_code == 429
    assert _is_retryable(exc) is True


def test_is_retryable_false_for_genuine_400() -> None:
    response = httpx.Response(status_code=400, request=httpx.Request("GET", "https://example.com"))
    exc = BadRequestError(
        message="malformed request", model="m", llm_provider="test", response=response
    )
    assert _is_retryable(exc) is False


def test_is_retryable_true_for_5xx_status_code_regardless_of_type() -> None:
    exc = Exception("upstream error")
    exc.status_code = 503  # type: ignore[attr-defined]
    assert _is_retryable(exc) is True
