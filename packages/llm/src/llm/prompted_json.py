"""Structured output via plain-text completion + robust parsing, never via
response_format/tool-calling. Some providers (confirmed: both Groq models in this
project's registry, groq/llama-3.1-8b-instant and groq/llama-3.3-70b-versatile, per
litellm.supports_response_schema) don't support native structured output — LiteLLM
silently rewrites CompletionRequest.response_model into a forced tool call for those
models, and small models calling that tool unreliably return Groq's
`tool_use_failed` error. complete_json() sidesteps that entirely: it never sets
response_format, so no provider in a tier's fallback chain is ever asked to invoke a
tool. This is additive — the default response_model path (llm.gateway.Gateway.complete)
is untouched and still used by nodes that don't need this."""

import json

from pydantic import BaseModel

from llm.gateway import Gateway
from llm.models import CompletionRequest, Message, Tier


class PromptedJsonError(Exception):
    """Raised when a prompted-JSON completion fails to parse into the target schema
    even after one repair retry."""


def extract_json_object(text: str) -> str:
    """Locates the JSON object in a raw model response: finds the first '{' and
    scans forward with a brace-depth counter (respecting quoted-string content, so a
    literal '{'/'}' inside a JSON string value doesn't miscount) to find its true
    matching close. Robust to markdown fences and leading/trailing prose or
    wrapper text around the object — the case this was built for is a Groq
    function-call-shaped response that still contains the real JSON inside it."""
    start = text.find("{")
    if start == -1:
        raise PromptedJsonError(f"no JSON object found in model output: {text!r}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise PromptedJsonError(f"unbalanced JSON object in model output: {text!r}")


_INSTRUCTION_TEMPLATE = (
    "Respond with ONLY a single JSON object matching this schema — no markdown "
    "code fences, no explanation, no other text:\n{schema}"
)

_REPAIR_TEMPLATE = (
    "Your previous reply was not valid JSON matching the schema:\n{previous}\n\n"
    "Parse error: {error}\n\n"
    "Reply again with ONLY the corrected JSON object — no markdown fences, no "
    "other text."
)


async def complete_json[T: BaseModel](
    gateway: Gateway,
    tier: Tier,
    messages: list[Message],
    response_model: type[T],
    max_tokens: int = 1024,
) -> T:
    """Gets a structured T via plain-text completion + our own parse, never via
    response_format. Every gateway.complete() call here still goes through the
    gateway's full retry/budget-guard/cache/fallback-chain machinery — only the
    "how do we get JSON out" layer is different. Real gateway failures
    (AllProvidersExhausted, BudgetExceeded) propagate immediately, not consumed by
    a repair attempt; only parse/validation failures trigger the one repair retry."""
    schema = json.dumps(response_model.model_json_schema())
    working_messages = [
        *messages,
        Message(role="user", content=_INSTRUCTION_TEMPLATE.format(schema=schema)),
    ]

    last_error: Exception | None = None
    for _attempt in range(2):  # initial attempt + exactly one repair retry
        request = CompletionRequest(tier=tier, messages=working_messages, max_tokens=max_tokens)
        result = await gateway.complete(request)
        try:
            json_str = extract_json_object(result.text)
            return response_model.model_validate_json(json_str)
        except Exception as exc:  # noqa: BLE001 — any parse/validation failure triggers repair
            last_error = exc
            working_messages = [
                *working_messages,
                Message(role="assistant", content=result.text),
                Message(
                    role="user",
                    content=_REPAIR_TEMPLATE.format(previous=result.text, error=str(exc)),
                ),
            ]

    raise PromptedJsonError(
        f"model output did not parse into {response_model.__name__} after a repair "
        f"retry: {last_error}"
    ) from last_error
