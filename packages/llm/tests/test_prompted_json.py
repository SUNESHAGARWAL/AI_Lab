from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import make_response
from fakeredis import FakeAsyncRedis
from llm.gateway import Gateway
from llm.models import Message, Tier
from llm.prompted_json import PromptedJsonError, complete_json, extract_json_object
from pydantic import BaseModel, ConfigDict


class _Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    supports: bool
    reason: str | None = None


def _sequenced_completion_fn(
    texts: list[str],
) -> tuple[Callable[..., Awaitable[SimpleNamespace]], list[dict[str, Any]]]:
    """Returns a completion_fn that yields `texts` in order, one per call — for
    proving a repair-retry actually happens on the second call, not the first."""
    calls: list[dict[str, Any]] = []

    async def _fn(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return make_response(texts[len(calls) - 1])

    return _fn, calls


def _gateway(completion_fn: Callable[..., Awaitable[SimpleNamespace]]) -> Gateway:
    return Gateway(redis_client=FakeAsyncRedis(), completion_fn=completion_fn)


# --- extract_json_object ---------------------------------------------------------


def test_extract_json_object_plain_json() -> None:
    assert extract_json_object('{"supports": true}') == '{"supports": true}'


def test_extract_json_object_strips_markdown_fences() -> None:
    text = '```json\n{"supports": true, "reason": "matches"}\n```'
    assert extract_json_object(text) == '{"supports": true, "reason": "matches"}'


def test_extract_json_object_strips_groq_function_call_blob() -> None:
    # Shape of what a small Groq model can emit instead of a clean tool call: prose
    # plus a function-call-looking wrapper around the real JSON.
    text = (
        "I'll call the function to report my verdict.\n"
        '<function=json_tool_call>{"supports": false, "reason": "unrelated"}'
        "</function>\nDone."
    )
    assert extract_json_object(text) == '{"supports": false, "reason": "unrelated"}'


def test_extract_json_object_handles_braces_inside_string_values() -> None:
    text = '{"supports": true, "reason": "the chunk says {x} directly"}'
    assert extract_json_object(text) == text


def test_extract_json_object_raises_on_no_json() -> None:
    with pytest.raises(PromptedJsonError):
        extract_json_object("no json here at all")


def test_extract_json_object_raises_on_unbalanced_braces() -> None:
    with pytest.raises(PromptedJsonError):
        extract_json_object('{"supports": true')


# --- complete_json -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_json_succeeds_on_first_attempt() -> None:
    completion_fn, calls = _sequenced_completion_fn(['{"supports": true, "reason": "ok"}'])
    gateway = _gateway(completion_fn)

    result = await complete_json(
        gateway, Tier.FAST, [Message(role="user", content="judge this")], _Verdict
    )

    assert result == _Verdict(supports=True, reason="ok")
    assert len(calls) == 1
    assert calls[0]["response_format"] is None  # never requests tool-calling


@pytest.mark.asyncio
async def test_complete_json_succeeds_on_repair_retry() -> None:
    completion_fn, calls = _sequenced_completion_fn(
        ["not json at all", '{"supports": true, "reason": "fixed"}']
    )
    gateway = _gateway(completion_fn)

    result = await complete_json(
        gateway, Tier.FAST, [Message(role="user", content="judge this")], _Verdict
    )

    assert result == _Verdict(supports=True, reason="fixed")
    assert len(calls) == 2
    repair_messages = calls[1]["messages"]
    assert any("not json at all" in m["content"] for m in repair_messages)
    assert any("corrected JSON" in m["content"] for m in repair_messages)


@pytest.mark.asyncio
async def test_complete_json_raises_after_repair_retry_also_fails() -> None:
    completion_fn, calls = _sequenced_completion_fn(["still not json", "still not json either"])
    gateway = _gateway(completion_fn)

    with pytest.raises(PromptedJsonError):
        await complete_json(
            gateway, Tier.FAST, [Message(role="user", content="judge this")], _Verdict
        )

    assert len(calls) == 2  # exactly initial attempt + one repair retry, no more
