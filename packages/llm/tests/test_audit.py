import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import llm.gateway as gateway_module
import pytest
from conftest import make_response
from core.testing import FakeEmbedder
from fakeredis import FakeAsyncRedis
from litellm.exceptions import BadRequestError
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
from llm.audit import AttemptRecorder
from llm.config import GatewaySettings
from llm.gateway import Gateway, _extract_status_code
from llm.models import CompletionRequest, Message, Tier
from llm.registry import ProviderModel, TierRegistry
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from pydantic import BaseModel
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


# --- Gateway integration: one record per billed attempt (1.1, 1.3, 1.6, 1.7, 1.8) ---

P1 = ProviderModel(provider="p1", model="p1/model", max_concurrency=4)
P2 = ProviderModel(provider="p2", model="p2/model", max_concurrency=4)


class _Answer(BaseModel):
    value: str


def _gateway(
    redis: FakeAsyncRedis,
    completion_fn: Callable[..., Awaitable[Any]],
    *providers: ProviderModel,
    audit_trace: Path | None,
    retries: int = 2,
    embedder: FakeEmbedder | None = None,
) -> Gateway:
    settings = GatewaySettings(
        audit_trace=str(audit_trace) if audit_trace is not None else None,
        same_provider_retry_attempts=retries,
        retry_backoff_initial_seconds=0,
        retry_backoff_max_seconds=0,
    )
    return Gateway(
        settings=settings,
        registry=TierRegistry({tier: list(providers) for tier in Tier}),
        redis_client=redis,
        embedder=embedder,
        completion_fn=completion_fn,
    )


def _litellm_429(model: str) -> LiteLLMRateLimitError:
    return LiteLLMRateLimitError(
        message=ERROR_SENTINEL, llm_provider="test", model=model, headers={"retry-after": "0"}
    )


def _402(model: str) -> BadRequestError:
    response = httpx.Response(status_code=402, request=httpx.Request("POST", "https://x.invalid"))
    return BadRequestError(
        message=ERROR_SENTINEL, model=model, llm_provider="test", response=response
    )


def _scripted(*outcomes: object) -> Callable[..., Awaitable[Any]]:
    """A fake completion_fn that raises or returns the next scripted outcome per call."""
    queue = list(outcomes)

    async def _completion_fn(**kwargs: Any) -> Any:
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return _completion_fn


def _user_request(**kwargs: Any) -> CompletionRequest:
    return CompletionRequest(
        tier=Tier.FAST, messages=[Message(role="user", content=USER_SENTINEL)], **kwargs
    )


def _attr(span: dict[str, Any], name: str) -> Any:
    return span["attributes"].get(name)


def _call_id(span: dict[str, Any]) -> str:
    return f"{span['context']['trace_id']}:{span['context']['span_id']}"


async def test_gateway_429_then_success_writes_two_linked_records(
    tmp_path: Path, fake_redis: FakeAsyncRedis
) -> None:
    target = tmp_path / "attempts.jsonl"
    gateway = _gateway(
        fake_redis,
        _scripted(_litellm_429("p1/model"), make_response("ok")),
        P1,
        audit_trace=target,
    )

    result = await gateway.complete(_user_request())

    assert result.text == "ok"
    assert result.retry_count == 1
    first, second = _spans(target)
    assert _attr(first, "llm_audit.attempt") == 1
    assert _attr(first, "llm_audit.retry_of") is None
    assert _attr(first, "error.type") == "RateLimitError"
    assert _attr(first, "http.response.status_code") == 429
    assert first["status"]["status_code"] == "ERROR"
    assert _attr(second, "llm_audit.attempt") == 2
    assert _attr(second, "llm_audit.retry_of") == _call_id(first)
    assert second["status"]["status_code"] == "UNSET"
    assert _attr(second, "gen_ai.usage.input_tokens") == 10
    assert _attr(second, "gen_ai.usage.output_tokens") == 5
    assert "SENTINEL" not in target.read_text(encoding="utf-8")


async def test_gateway_retry_exhaustion_then_fallback_continues_numbering(
    tmp_path: Path, fake_redis: FakeAsyncRedis
) -> None:
    target = tmp_path / "attempts.jsonl"
    gateway = _gateway(
        fake_redis,
        _scripted(_litellm_429("p1/model"), _litellm_429("p1/model"), make_response("from p2")),
        P1,
        P2,
        audit_trace=target,
    )

    result = await gateway.complete(_user_request())

    assert result.provider == "p2"
    spans = _spans(target)
    assert [_attr(s, "llm_audit.attempt") for s in spans] == [1, 2, 3]
    assert [_attr(s, "gen_ai.provider.name") for s in spans] == ["p1", "p1", "p2"]
    assert [_attr(s, "llm_audit.retry_of") for s in spans] == [
        None,
        _call_id(spans[0]),
        _call_id(spans[1]),
    ]
    assert [s["status"]["status_code"] for s in spans] == ["ERROR", "ERROR", "UNSET"]


async def test_gateway_provider_unavailable_then_fallback_continues_numbering(
    tmp_path: Path, fake_redis: FakeAsyncRedis
) -> None:
    target = tmp_path / "attempts.jsonl"
    gateway = _gateway(
        fake_redis,
        _scripted(_402("p1/model"), make_response("from p2")),
        P1,
        P2,
        audit_trace=target,
    )

    result = await gateway.complete(_user_request())

    assert result.provider == "p2"
    first, second = _spans(target)
    assert _attr(first, "error.type") == "BadRequestError"
    assert _attr(first, "http.response.status_code") == 402
    assert _attr(second, "llm_audit.attempt") == 2
    assert _attr(second, "llm_audit.retry_of") == _call_id(first)
    assert "SENTINEL" not in target.read_text(encoding="utf-8")


async def test_gateway_parse_failure_is_a_failed_record_with_usage(
    tmp_path: Path, fake_redis: FakeAsyncRedis
) -> None:
    target = tmp_path / "attempts.jsonl"
    gateway = _gateway(
        fake_redis,
        _scripted(
            make_response("not json", prompt_tokens=30, completion_tokens=7),
            make_response('{"value": "42"}'),
        ),
        P1,
        audit_trace=target,
    )

    result = await gateway.complete(_user_request(response_model=_Answer))

    assert isinstance(result.parsed, _Answer)
    failed, ok = _spans(target)
    assert _attr(failed, "error.type") == "ValidationError"
    assert failed["status"]["status_code"] == "ERROR"
    assert _attr(failed, "gen_ai.usage.input_tokens") == 30
    assert _attr(failed, "gen_ai.usage.output_tokens") == 7
    assert "http.response.status_code" not in failed["attributes"]
    assert _attr(ok, "llm_audit.retry_of") == _call_id(failed)
    assert ok["status"]["status_code"] == "UNSET"


async def test_gateway_cache_hit_writes_no_record(
    tmp_path: Path, fake_redis: FakeAsyncRedis
) -> None:
    target = tmp_path / "attempts.jsonl"
    calls: list[dict[str, Any]] = []

    async def completion_fn(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return make_response("cached answer")

    gateway = _gateway(fake_redis, completion_fn, P1, audit_trace=target, embedder=FakeEmbedder())
    request = _user_request()

    first = await gateway.complete(request)
    lines_after_first = len(_spans(target))
    second = await gateway.complete(request)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1
    assert lines_after_first == 1
    assert len(_spans(target)) == 1


async def test_gateway_cancelled_call_writes_a_failed_record_and_propagates(
    tmp_path: Path, fake_redis: FakeAsyncRedis
) -> None:
    target = tmp_path / "attempts.jsonl"
    called = asyncio.Event()

    async def completion_fn(**kwargs: Any) -> Any:
        called.set()
        await asyncio.Event().wait()  # never returns; the request is cancelled mid-call

    gateway = _gateway(fake_redis, completion_fn, P1, audit_trace=target)
    task = asyncio.create_task(gateway.complete(_user_request()))
    await called.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    [span] = _spans(target)
    assert _attr(span, "error.type") == "CancelledError"
    assert span["status"]["status_code"] == "ERROR"
    assert _attr(span, "llm_audit.attempt") == 1


def _broken_inference(self: object, *args: object, **kwargs: object) -> object:
    raise RuntimeError(ERROR_SENTINEL)


def _broken_end(self: object, *args: object, **kwargs: object) -> None:
    raise RuntimeError(ERROR_SENTINEL)


@pytest.mark.parametrize("broken", ["start", "end"])
async def test_gateway_failing_recorder_leaves_the_result_unchanged(
    tmp_path: Path, fake_redis: FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    def _outcomes() -> Callable[..., Awaitable[Any]]:
        return _scripted(_litellm_429("p1/model"), make_response('{"value": "42"}'))

    baseline = await _gateway(FakeAsyncRedis(), _outcomes(), P1, audit_trace=None).complete(
        _user_request(response_model=_Answer)
    )

    if broken == "start":
        monkeypatch.setattr(TelemetryHandler, "inference", _broken_inference)
    else:
        monkeypatch.setattr(InferenceInvocation, "stop", _broken_end)
        monkeypatch.setattr(InferenceInvocation, "fail", _broken_end)
    gateway = _gateway(fake_redis, _outcomes(), P1, audit_trace=tmp_path / "attempts.jsonl")
    with capture_logs() as logs:
        result = await gateway.complete(_user_request(response_model=_Answer))

    assert result.model_dump(exclude={"latency_ms"}) == baseline.model_dump(exclude={"latency_ms"})
    audit_logs = [entry for entry in logs if entry["event"] == "llm.audit_trace_failed"]
    assert len(audit_logs) == 2
    assert "SENTINEL" not in repr(logs)


async def test_gateway_with_recording_off_runs_no_recorder_code(
    tmp_path: Path, fake_redis: FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("recorder code ran with recording off")

    monkeypatch.setattr(gateway_module, "open_audit_sink", _must_not_run)
    monkeypatch.setattr(gateway_module.AttemptRecorder, "__init__", _must_not_run)
    monkeypatch.setattr(gateway_module.AttemptRecorder, "start", _must_not_run)
    gateway = _gateway(
        fake_redis,
        _scripted(_litellm_429("p1/model"), make_response("ok")),
        P1,
        audit_trace=None,
    )

    result = await gateway.complete(_user_request())

    assert result.text == "ok"
    assert list(tmp_path.iterdir()) == []


# --- Synthetic contract fixture for the llm-audit OTel GenAI ingest (1.6, 1.9) ---
#
# The committed fixture is a gateway trace from fake completions only; it holds no real data.
# Trace and span ids and times are random per run, so by default the test only checks that a
# fresh trace has the committed structure. Regenerate after any change to the record contract:
#   AI_LAB_WRITE_FIXTURE=1 uv run pytest packages/llm/tests/test_audit.py -k contract_fixture
# then copy the file to llm-audit's tests/fixtures/ for its contract test.

CONTRACT_FIXTURE = Path(__file__).parent / "fixtures" / "ai_lab_attempts.jsonl"
# Models the llm-audit price snapshot prices, so the ingest leaves nothing unpriced.
DS_FLASH = ProviderModel(provider="deepseek", model="deepseek/deepseek-flash", max_concurrency=4)
DS_PRO = ProviderModel(provider="deepseek", model="deepseek/deepseek-v4-pro", max_concurrency=4)
# Fields that differ on every run: ids, links made of ids, times and the SDK's per-process id.
_VOLATILE_KEYS = ("context", "parent_id", "start_time", "end_time")
_VOLATILE_RESOURCE_KEYS = ("service.instance.id",)


def _contract_request(case: str, *, system: bool, feature: str | None) -> CompletionRequest:
    messages = [Message(role="user", content=f"{USER_SENTINEL} case {case}")]
    if system:
        messages.insert(0, Message(role="system", content=SYSTEM_SENTINEL))
    return CompletionRequest(
        tier=Tier.REASON, messages=messages, temperature=0.2, max_tokens=256, feature=feature
    )


def _400(model: str) -> BadRequestError:
    response = httpx.Response(status_code=400, request=httpx.Request("POST", "https://x.invalid"))
    return BadRequestError(
        message=ERROR_SENTINEL, model=model, llm_provider="test", response=response
    )


async def _write_contract_trace(target: Path, redis: FakeAsyncRedis) -> None:
    """Drives the real gateway through every attempt shape the field run can produce."""
    gateway = Gateway(
        settings=GatewaySettings(
            _env_file=None,  # type: ignore[call-arg]
            app_env="synthetic",
            audit_trace=str(target),
            same_provider_retry_attempts=2,
            retry_backoff_initial_seconds=0,
            retry_backoff_max_seconds=0,
        ),
        registry=TierRegistry({tier: [DS_FLASH, DS_PRO] for tier in Tier}),
        redis_client=redis,
        completion_fn=_scripted(
            # (a) success with a system message
            _response(model="deepseek-flash"),
            # (b) same-provider 429, then success
            _litellm_429(DS_FLASH.model),
            _response(model="deepseek-flash", cached=0, reasoning=0),
            # (c) retries exhausted on the first provider, then fallback to the second
            _litellm_429(DS_FLASH.model),
            _litellm_429(DS_FLASH.model),
            _response(model="deepseek-v4-pro", finish_reason="length"),
            # (d) a non-retryable error that ends the request
            _400(DS_FLASH.model),
            # (e) no system message and no feature
            _response(model="deepseek-flash", cached=None, reasoning=None),
        ),
    )
    a = await gateway.complete(_contract_request("a", system=True, feature="generator"))
    b = await gateway.complete(_contract_request("b", system=True, feature="planner"))
    c = await gateway.complete(_contract_request("c", system=True, feature="judge_metric"))
    with pytest.raises(BadRequestError):
        await gateway.complete(_contract_request("d", system=True, feature="critic"))
    e = await gateway.complete(_contract_request("e", system=False, feature=None))
    assert [a.provider, b.retry_count, c.model, e.provider] == [
        "deepseek",
        1,
        DS_PRO.model,
        "deepseek",
    ]


def _contract_shape(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each line without its run-specific fields; retry_of becomes the index of the named line."""
    index = {_call_id(span): i for i, span in enumerate(lines)}
    shapes = []
    for span in lines:
        shape = {k: v for k, v in span.items() if k not in _VOLATILE_KEYS}
        attributes = dict(span["attributes"])
        if "llm_audit.retry_of" in attributes:
            attributes["llm_audit.retry_of"] = index[attributes["llm_audit.retry_of"]]
        shape["attributes"] = attributes
        resource = span["resource"]["attributes"]
        shape["resource"] = {k: v for k, v in resource.items() if k not in _VOLATILE_RESOURCE_KEYS}
        shapes.append(shape)
    return shapes


async def test_contract_fixture_matches_a_fresh_gateway_trace(
    tmp_path: Path, fake_redis: FakeAsyncRedis
) -> None:
    target = tmp_path / "attempts.jsonl"
    await _write_contract_trace(target, fake_redis)

    raw = target.read_text(encoding="utf-8")
    for sentinel in ("SENTINEL", "statute", "article 5", "sk-abc", "case "):
        assert sentinel not in raw
    fresh = _spans(target)
    shapes = _contract_shape(fresh)
    assert [s["attributes"].get("llm_audit.feature") for s in shapes] == [
        "generator",
        "planner",
        "planner",
        "judge_metric",
        "judge_metric",
        "judge_metric",
        "critic",
        None,
    ]
    assert [s["attributes"]["llm_audit.attempt"] for s in shapes] == [1, 1, 2, 1, 2, 3, 1, 1]
    assert [s["attributes"].get("llm_audit.retry_of") for s in shapes] == [
        None, None, 1, None, 3, 4, None, None,
    ]  # fmt: skip
    assert [s["attributes"].get("error.type") for s in shapes] == [
        None, "RateLimitError", None, "RateLimitError", "RateLimitError", None,
        "BadRequestError", None,
    ]  # fmt: skip
    assert "llm_audit.prefix_hash" not in shapes[-1]["attributes"]

    if os.environ.get("AI_LAB_WRITE_FIXTURE") == "1":
        CONTRACT_FIXTURE.parent.mkdir(exist_ok=True)
        CONTRACT_FIXTURE.write_text(raw, encoding="utf-8")
    committed = [json.loads(line) for line in CONTRACT_FIXTURE.read_text().splitlines()]
    assert _contract_shape(committed) == shapes
