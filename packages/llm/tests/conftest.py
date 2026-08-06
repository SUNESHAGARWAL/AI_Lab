from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from core.testing import FakeEmbedder
from fakeredis import FakeAsyncRedis


@pytest_asyncio.fixture
async def fake_redis() -> FakeAsyncRedis:
    redis = FakeAsyncRedis()
    yield redis
    await redis.aclose()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_response(
    text: str, prompt_tokens: int = 10, completion_tokens: int = 5
) -> SimpleNamespace:
    """A `litellm.ModelResponse`-shaped stand-in: `.choices[0].message.content` and
    `.usage.{prompt,completion,total}_tokens`. No real LiteLLM class involved."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_completion_fn(
    text: str = "hello",
) -> tuple[Callable[..., Awaitable[SimpleNamespace]], list[dict[str, Any]]]:
    """Returns (fake completion_fn, list of call kwargs it was invoked with)."""
    calls: list[dict[str, Any]] = []

    async def _completion_fn(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return make_response(text)

    return _completion_fn, calls
