import json
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from evals.judge import DEFAULT_JUDGE_TIER, GatewayJudgeModel, judge_citation_support
from fakeredis import FakeAsyncRedis
from llm.config import GatewaySettings

from llm import AllProvidersExhausted, Gateway

# Local copy of packages/llm/tests/conftest.py's make_response/make_completion_fn
# pattern — not importable across package test dirs, so duplicated here, same as
# apps/api/tests/test_nodes.py does.


def _completion_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_completion_fn(
    text: str,
) -> tuple[Callable[..., Awaitable[SimpleNamespace]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    async def _fn(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion_response(text)

    return _fn, calls


def _gateway(
    completion_fn: Callable[..., Awaitable[SimpleNamespace]], **settings_kwargs: Any
) -> Gateway:
    return Gateway(
        settings=GatewaySettings(**settings_kwargs),
        redis_client=FakeAsyncRedis(),
        completion_fn=completion_fn,
    )


@pytest.mark.asyncio
async def test_gateway_judge_model_a_generate_returns_gateway_text() -> None:
    completion_fn, calls = _make_completion_fn("the judge's raw text response")
    model = GatewayJudgeModel(_gateway(completion_fn))

    result = await model.a_generate("some judge prompt")

    assert result == "the judge's raw text response"
    assert len(calls) == 1
    assert calls[0]["response_format"] is None  # deepeval prompts are plain text


def test_gateway_judge_model_get_model_name_reflects_tier() -> None:
    completion_fn, _ = _make_completion_fn("x")
    model = GatewayJudgeModel(_gateway(completion_fn), tier=DEFAULT_JUDGE_TIER)

    assert model.get_model_name() == f"gateway:{DEFAULT_JUDGE_TIER.value}"


def test_gateway_judge_model_generate_raises_not_implemented() -> None:
    completion_fn, _ = _make_completion_fn("x")
    model = GatewayJudgeModel(_gateway(completion_fn))

    with pytest.raises(NotImplementedError):
        model.generate("some prompt")


@pytest.mark.asyncio
async def test_judge_citation_support_parses_valid_verdict() -> None:
    payload = json.dumps({"supports": True, "reason": "the chunk directly states this"})
    completion_fn, calls = _make_completion_fn(payload)

    verdict = await judge_citation_support(_gateway(completion_fn), "chunk text", "answer text")

    assert verdict.supports is True
    assert verdict.reason == "the chunk directly states this"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_judge_citation_support_raises_on_malformed_response() -> None:
    completion_fn, _ = _make_completion_fn("not json at all")

    with pytest.raises(AllProvidersExhausted):
        await judge_citation_support(
            _gateway(completion_fn, same_provider_retry_attempts=1), "chunk text", "answer text"
        )
