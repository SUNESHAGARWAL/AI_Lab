from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Tier(StrEnum):
    """Nodes ask for a tier, never a model string — see `.claude/rules/llm-gateway.md`."""

    FAST = "fast"
    REASON = "reason"
    BULK = "bulk"
    LOCAL = "local"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    """A single gateway call. `response_model` opts into structured output; a plain
    completion (no schema) leaves it `None` and `CompletionResult.parsed` stays `None`."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tier: Tier
    messages: list[Message] = Field(min_length=1)
    temperature: float = 0.0
    max_tokens: int = Field(default=1024, gt=0)
    response_model: type[BaseModel] | None = None
    user_uploaded_content: bool = False


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str
    parsed: BaseModel | None
    usage: Usage
    provider: str
    model: str
    tier: Tier
    cache_hit: bool
    cache_layer: Literal["exact", "semantic"] | None
    estimated_cost_usd: float
    latency_ms: float
    retry_count: int
