"""Audit trace sink: one content-free GenAI inference span per model call attempt, as JSON lines.

The sink owns a private TracerProvider, so the app's own spans (the per-request gateway span) never
reach the audit trace, and the global tracer provider is never touched. The GenAI handler from
`opentelemetry-util-genai` emits the span names and token attributes; this module only chooses where
the lines go and refuses to start when either of that library's content switches is on.
"""

import functools
import json
import os
import sys
from pathlib import Path

from opentelemetry._logs import NoOpLoggerProvider
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ContentCapturingMode

STDOUT_TARGET = "stdout"
STDOUT_KEY = "llm_audit_span"


class ContentCaptureEnabled(RuntimeError):
    """The GenAI content-capture environment would put prompt or completion text in the trace."""


def _file_line(span: ReadableSpan) -> str:
    # The compact span JSON that `llm-audit ingest` reads, one per line.
    line: str = span.to_json(indent=None)
    return line + "\n"


def _stdout_line(span: ReadableSpan) -> str:
    # One JSON string under one key, so Cloud Logging keeps the span line intact in jsonPayload.
    return json.dumps({STDOUT_KEY: span.to_json(indent=None)}) + "\n"


def _refuse_content_capture() -> None:
    # Same parsing as the library (strip, upper-case, empty means NO_CONTENT); unknown values,
    # which the library downgrades to NO_CONTENT with a warning, are refused here too.
    mode = os.environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "").strip()
    if mode and mode.upper() != ContentCapturingMode.NO_CONTENT.name:
        raise ContentCaptureEnabled(
            f"{OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT} must be unset or NO_CONTENT "
            "while the audit trace is on"
        )
    # The library treats an empty hook name as unset.
    if os.environ.get(OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK):
        raise ContentCaptureEnabled(
            f"{OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK} must be unset "
            "while the audit trace is on"
        )


@functools.cache
def open_audit_sink(target: str) -> TelemetryHandler:
    """target: a file path (opened for append, parent must exist) or the literal "stdout".

    Raises ContentCaptureEnabled if OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is set to any
    value other than NO_CONTENT, or if OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK is set.
    Raises FileNotFoundError if the parent folder of a file target is missing.
    Cached per target, so every gateway in the process shares one sink per target.
    """
    _refuse_content_capture()
    if target == STDOUT_TARGET:
        exporter = ConsoleSpanExporter(out=sys.stdout, formatter=_stdout_line)
    else:
        # Kept open for the life of the process; the exporter flushes after every span.
        out = Path(target).open("a", encoding="utf-8")  # noqa: SIM115
        exporter = ConsoleSpanExporter(out=out, formatter=_file_line)
    # ALWAYS_ON: every attempt is recorded, even under an unsampled app span or OTEL_TRACES_SAMPLER.
    provider = TracerProvider(sampler=ALWAYS_ON)
    # Synchronous: each span is written and flushed before stop() or fail() returns.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TelemetryHandler(
        tracer_provider=provider,
        meter_provider=NoOpMeterProvider(),
        logger_provider=NoOpLoggerProvider(),
    )
