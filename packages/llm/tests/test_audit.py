import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from llm.audit import AttemptRecorder
from llm.gateway import _extract_status_code
from llm.models import CompletionRequest, Message, Tier
from llm.registry import ProviderModel
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK,
)
from structlog.testing import capture_logs
from telemetry.audit_trace import open_audit_sink

# Distinctive strings that must never reach the trace file.
SYSTEM_SENTINEL = "SYSTEM-SENTINEL-7f3a quote the statute verbatim"
USER_SENTINEL = "USER-SENTINEL-91bc what does article 5 forbid"
ERROR_SENTINEL = "PROVIDER-ERROR-SENTINEL-c0de quota for key sk-abc exceeded"

PROVIDER = ProviderModel(provider="deepseek", model="deepseek/deepseek-flash")


@pytest.fixture(autouse=True)
def _fresh_sinks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, raising=False)
    monkeypatch.delenv(OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK, raising=False)
    open_audit_sink.cache_clear()
    yield
    open_audit_sink.cache_clear()


class RateLimitError(Exception):
    """Same short name and status attribute as LiteLLM's, without importing it."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 429


def _recorder(target: Path, env: str = "dev") -> AttemptRecorder:
    return AttemptRecorder(open_audit_sink(str(target)), env=env, status_code=_extract_status_code)


def _request(*, system: bool = True, feature: str | None = "generator") -> CompletionRequest:
    messages = [Message(role="user", content=USER_SENTINEL)]
    if system:
        messages = [
            Message(role="system", content=SYSTEM_SENTINEL),
            Message(role="system", content=SYSTEM_SENTINEL + " second"),
            *messages,
            Message(role="system", content="late system part, not prefix"),
        ]
    return CompletionRequest(
        tier=Tier.REASON,
        messages=messages,
        temperature=0.3,
        max_tokens=512,
        feature=feature,
    )


def _response(
    *,
    model: str | None = "deepseek-v4-flash",
    finish_reason: str | None = "stop",
    cached: int | None = 64,
    reasoning: int | None = 20,
    details: bool = True,
) -> SimpleNamespace:
    """Shaped like LiteLLM's ModelResponse for DeepSeek; the fields the recorder reads only."""
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=45, total_tokens=165)
    if details:
        usage.prompt_tokens_details = SimpleNamespace(cached_tokens=cached)
        usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=reasoning)
    choice = SimpleNamespace(
        message=SimpleNamespace(content="COMPLETION-SENTINEL-55aa"), finish_reason=finish_reason
    )
    response = SimpleNamespace(choices=[choice], usage=usage)
    if model is not None:
        response.model = model
    return response


def _spans(target: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]


def _sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_success_records_every_contract_attribute(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    request = _request()
    handle = _recorder(target, env="prod").start(
        provider=PROVIDER, request=request, attempt=2, retry_of="0x" + "a" * 32 + ":0x" + "b" * 16
    )
    handle.set_response(_response())
    handle.succeed()

    [span] = _spans(target)
    prefix = [m.model_dump() for m in request.messages[:2]]
    assert span["attributes"] == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "deepseek",
        "gen_ai.request.model": "deepseek/deepseek-flash",
        "gen_ai.request.max_tokens": 512,
        "gen_ai.request.temperature": 0.3,
        "gen_ai.response.model": "deepseek-v4-flash",
        "gen_ai.response.finish_reasons": ["stop"],
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 45,
        "gen_ai.usage.cache_read.input_tokens": 64,
        "gen_ai.usage.reasoning.output_tokens": 20,
        "llm_audit.attempt": 2,
        "llm_audit.retry_of": "0x" + "a" * 32 + ":0x" + "b" * 16,
        "llm_audit.feature": "generator",
        "llm_audit.env": "prod",
        "llm_audit.prompt_hash": _sha([m.model_dump() for m in request.messages]),
        "llm_audit.prefix_hash": _sha(prefix),
        "llm_audit.prefix_tokens": litellm.token_counter(
            model="deepseek/deepseek-flash", messages=prefix
        ),
    }
    assert span["attributes"]["llm_audit.prefix_tokens"] > 0
    assert span["status"]["status_code"] == "UNSET"
    assert span["start_time"] and span["end_time"]


def test_call_id_is_the_adapter_form_of_the_line_context(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.succeed()

    [span] = _spans(target)
    assert handle.call_id == f"{span['context']['trace_id']}:{span['context']['span_id']}"
    trace_id, span_id = handle.call_id.split(":")
    assert len(trace_id) == 34 and trace_id.startswith("0x")
    assert len(span_id) == 18 and span_id.startswith("0x")


def test_no_message_system_or_completion_text_reaches_the_file(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    recorder = _recorder(target)
    ok = recorder.start(provider=PROVIDER, request=_request(), attempt=1, retry_of=None)
    ok.set_response(_response())
    ok.succeed()
    bad = recorder.start(provider=PROVIDER, request=_request(), attempt=2, retry_of=ok.call_id)
    bad.fail(RateLimitError(ERROR_SENTINEL))

    raw = target.read_text(encoding="utf-8")
    for sentinel in ("SENTINEL", "statute", "article 5", "late system part", "sk-abc"):
        assert sentinel not in raw


def test_absent_usage_details_leave_their_attributes_absent(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.set_response(_response(model=None, finish_reason=None, details=False))
    handle.succeed()

    [span] = _spans(target)
    attrs = span["attributes"]
    for absent in (
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.reasoning.output_tokens",
    ):
        assert absent not in attrs
    assert attrs["gen_ai.usage.input_tokens"] == 120
    assert attrs["gen_ai.usage.output_tokens"] == 45


def test_details_object_without_counts_leaves_them_absent(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.set_response(_response(cached=None, reasoning=None))
    handle.succeed()

    [span] = _spans(target)
    assert "gen_ai.usage.cache_read.input_tokens" not in span["attributes"]
    assert "gen_ai.usage.reasoning.output_tokens" not in span["attributes"]


def test_reported_zero_cache_and_reasoning_are_kept_as_zero(tmp_path: Path) -> None:
    # A reported 0 is a measurement ("no cache hit"), not a missing field (1.5).
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.set_response(_response(cached=0, reasoning=0))
    handle.succeed()

    [span] = _spans(target)
    assert span["attributes"]["gen_ai.usage.cache_read.input_tokens"] == 0
    assert span["attributes"]["gen_ai.usage.reasoning.output_tokens"] == 0


def test_no_feature_no_system_message_and_first_attempt_omit_their_attributes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "attempts.jsonl"
    request = _request(system=False, feature=None)
    handle = _recorder(target).start(provider=PROVIDER, request=request, attempt=1, retry_of=None)
    handle.succeed()

    [span] = _spans(target)
    attrs = span["attributes"]
    for absent in (
        "llm_audit.feature",
        "llm_audit.prefix_hash",
        "llm_audit.prefix_tokens",
        "llm_audit.retry_of",
    ):
        assert absent not in attrs
    assert attrs["llm_audit.attempt"] == 1
    assert attrs["llm_audit.prompt_hash"] == _sha([m.model_dump() for m in request.messages])


def test_no_tenant_user_ip_server_or_response_id_attribute(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.set_response(_response())
    handle.succeed()

    [span] = _spans(target)
    for name in span["attributes"]:
        assert not name.startswith(("server.", "client.", "user.", "enduser."))
        assert name not in ("gen_ai.response.id", "llm_audit.tenant", "llm_audit.user_id_hash")


def test_failure_records_short_type_status_code_and_redacted_status(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.fail(RateLimitError(ERROR_SENTINEL))

    [span] = _spans(target)
    assert span["attributes"]["error.type"] == "RateLimitError"
    assert span["attributes"]["http.response.status_code"] == 429
    assert span["status"]["status_code"] == "ERROR"
    assert span["status"]["description"] == "RateLimitError"
    assert span["events"] == []
    assert "gen_ai.usage.input_tokens" not in span["attributes"]


def test_failure_without_status_code_omits_http_status(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.fail(ValueError(ERROR_SENTINEL))

    [span] = _spans(target)
    assert span["attributes"]["error.type"] == "ValueError"
    assert "http.response.status_code" not in span["attributes"]
    assert ERROR_SENTINEL not in json.dumps(span)


def test_failure_after_response_keeps_billed_usage(tmp_path: Path) -> None:
    # A structured-output parse failure: the call was billed, then the attempt failed.
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    handle.set_response(_response())
    handle.fail(ValueError(ERROR_SENTINEL))

    [span] = _spans(target)
    assert span["attributes"]["gen_ai.usage.input_tokens"] == 120
    assert span["attributes"]["error.type"] == "ValueError"
    assert span["status"]["status_code"] == "ERROR"


def test_retry_chain_carries_attempt_and_previous_call_id(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    recorder = _recorder(target)
    first = recorder.start(provider=PROVIDER, request=_request(), attempt=1, retry_of=None)
    first.fail(RateLimitError(ERROR_SENTINEL))
    second = recorder.start(
        provider=PROVIDER, request=_request(), attempt=2, retry_of=first.call_id
    )
    second.set_response(_response())
    second.succeed()

    one, two = _spans(target)
    assert one["attributes"]["llm_audit.attempt"] == 1
    assert two["attributes"]["llm_audit.attempt"] == 2
    assert two["attributes"]["llm_audit.retry_of"] == first.call_id
    assert first.call_id != second.call_id


class _BrokenHandler:
    def inference(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError(ERROR_SENTINEL)


def test_recorder_errors_are_logged_without_content_and_never_raised() -> None:
    recorder = AttemptRecorder(
        _BrokenHandler(),  # type: ignore[arg-type]
        env="dev",
        status_code=_extract_status_code,
    )
    with capture_logs() as logs:
        handle = recorder.start(provider=PROVIDER, request=_request(), attempt=1, retry_of=None)
        handle.set_response(_response())
        handle.succeed()
        handle.fail(RateLimitError(ERROR_SENTINEL))

    assert handle.call_id is None
    assert logs and all(entry["event"] == "llm.audit_trace_failed" for entry in logs)
    assert "SENTINEL" not in repr(logs)


def test_failing_status_code_lookup_still_ends_the_record(tmp_path: Path) -> None:
    def _boom(exc: BaseException) -> int | None:
        raise RuntimeError(ERROR_SENTINEL)

    target = tmp_path / "attempts.jsonl"
    recorder = AttemptRecorder(open_audit_sink(str(target)), env="dev", status_code=_boom)
    with capture_logs() as logs:
        handle = recorder.start(provider=PROVIDER, request=_request(), attempt=1, retry_of=None)
        handle.fail(RateLimitError(ERROR_SENTINEL))

    [span] = _spans(target)
    assert span["attributes"]["error.type"] == "RateLimitError"
    assert "http.response.status_code" not in span["attributes"]
    assert [entry["event"] for entry in logs] == ["llm.audit_trace_failed"]
    assert "SENTINEL" not in repr(logs)


def test_malformed_response_is_logged_and_the_record_still_ends(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handle = _recorder(target).start(
        provider=PROVIDER, request=_request(), attempt=1, retry_of=None
    )
    with capture_logs() as logs:
        # A set is truthy but not indexable, so reading choices[0] raises inside the recorder.
        handle.set_response(SimpleNamespace(choices={"x"}, usage=None))
        handle.succeed()

    [span] = _spans(target)
    assert [entry["event"] for entry in logs] == ["llm.audit_trace_failed"]
    assert span["status"]["status_code"] == "UNSET"
    assert "gen_ai.usage.input_tokens" not in span["attributes"]
    assert "SENTINEL" not in repr(logs)
