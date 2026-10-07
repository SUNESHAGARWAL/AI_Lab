import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from telemetry.audit_trace import (
    ContentCaptureEnabled,
    _file_line,
    _stdout_line,
    open_audit_sink,
)


@pytest.fixture(autouse=True)
def _fresh_sinks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test builds its own sink and starts from a clean content-capture environment."""
    monkeypatch.delenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, raising=False)
    monkeypatch.delenv(OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK, raising=False)
    open_audit_sink.cache_clear()
    yield
    open_audit_sink.cache_clear()


def _one_span(handler: TelemetryHandler) -> None:
    invocation = handler.inference("deepseek", request_model="deepseek-chat", operation_name="chat")
    invocation.input_tokens = 12
    invocation.output_tokens = 3
    invocation.stop()


def test_file_target_writes_one_parseable_line_per_span(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    _one_span(open_audit_sink(str(target)))

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    span = json.loads(lines[0])
    assert span["context"]["trace_id"].startswith("0x")
    assert span["context"]["span_id"].startswith("0x")
    assert span["name"] == "chat deepseek-chat"
    assert span["attributes"]["gen_ai.usage.input_tokens"] == 12


def test_file_target_appends(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    target.write_text('{"earlier": true}\n', encoding="utf-8")
    handler = open_audit_sink(str(target))
    _one_span(handler)
    _one_span(handler)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"earlier": True}


def test_stdout_target_wraps_one_span_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    _one_span(open_audit_sink("stdout"))

    out_lines = capsys.readouterr().out.splitlines()
    assert len(out_lines) == 1
    envelope = json.loads(out_lines[0])
    assert list(envelope) == ["llm_audit_span"]
    inner = envelope["llm_audit_span"]
    assert isinstance(inner, str)
    span = json.loads(inner)
    assert span["name"] == "chat deepseek-chat"
    assert span["context"]["trace_id"].startswith("0x")


def test_stdout_envelope_round_trips_to_the_file_line() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("chat deepseek-chat") as live:
        live.set_attribute("gen_ai.usage.input_tokens", 12)
    (span,) = exporter.get_finished_spans()

    file_line = _file_line(span)
    stdout_line = _stdout_line(span)
    assert file_line.endswith("\n") and file_line.count("\n") == 1
    assert stdout_line.endswith("\n") and stdout_line.count("\n") == 1
    assert json.loads(stdout_line)["llm_audit_span"] + "\n" == file_line


@pytest.mark.parametrize("value", ["SPAN_ONLY", "span_and_event", " EVENT_ONLY ", "bogus"])
def test_content_capture_env_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, value)
    with pytest.raises(ContentCaptureEnabled):
        open_audit_sink(str(tmp_path / "attempts.jsonl"))
    assert not (tmp_path / "attempts.jsonl").exists()


@pytest.mark.parametrize("value", ["", "NO_CONTENT", "no_content", " No_Content "])
def test_no_content_values_are_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, value)
    assert isinstance(open_audit_sink(str(tmp_path / "attempts.jsonl")), TelemetryHandler)


def test_completion_hook_env_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK, "upload")
    with pytest.raises(ContentCaptureEnabled):
        open_audit_sink(str(tmp_path / "attempts.jsonl"))


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_audit_sink(str(tmp_path / "missing" / "attempts.jsonl"))


def test_global_tracer_spans_never_reach_the_file(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handler = open_audit_sink(str(target))
    global_provider_before = trace.get_tracer_provider()

    with trace.get_tracer("gateway").start_as_current_span("gateway.complete"):
        _one_span(handler)
    with trace.get_tracer("gateway").start_as_current_span("gateway.other"):
        pass

    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["name"] for line in lines] == ["chat deepseek-chat"]
    assert trace.get_tracer_provider() is global_provider_before


def test_same_target_returns_the_same_handler(tmp_path: Path) -> None:
    target = str(tmp_path / "attempts.jsonl")
    assert open_audit_sink(target) is open_audit_sink(target)
    assert open_audit_sink(target) is not open_audit_sink(str(tmp_path / "other.jsonl"))


def test_sampler_env_does_not_drop_audit_spans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    target = tmp_path / "attempts.jsonl"
    _one_span(open_audit_sink(str(target)))

    assert len(target.read_text(encoding="utf-8").splitlines()) == 1


def test_unsampled_parent_does_not_drop_audit_spans(tmp_path: Path) -> None:
    target = tmp_path / "attempts.jsonl"
    handler = open_audit_sink(str(target))
    app_provider = TracerProvider(sampler=ParentBased(ALWAYS_OFF))

    with app_provider.get_tracer("gateway").start_as_current_span("gateway.complete"):
        _one_span(handler)

    assert len(target.read_text(encoding="utf-8").splitlines()) == 1
