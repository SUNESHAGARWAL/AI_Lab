"""Redis-backed response cache: an exact-match layer in front of a semantic layer.

Exact match is a plain key/value lookup — cheap, always tried first. The semantic
layer is a hand-rolled brute-force cosine-similarity scan over a small capped
candidate list, not `redisvl.extensions.cache.llm.SemanticCache` — that library needs
RediSearch/the Redis Query Engine, which the pinned `redis:7-alpine` compose image
doesn't have. At this project's cache size a bounded brute-force scan is genuinely
sufficient; this is a deliberate, documented exception to "use the library."
"""

import hashlib
import json
import math

from core.ports import Embedder
from redis.asyncio import Redis

from llm.config import GatewaySettings
from llm.models import CompletionRequest, CompletionResult, Tier, Usage

_EXACT_PREFIX = "llm:cache:exact"
_SEMANTIC_PREFIX = "llm:cache:semantic"


def _cache_key(request: CompletionRequest) -> str:
    payload = {
        "tier": request.tier.value,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "response_model": request.response_model.__name__ if request.response_model else None,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def _serialize(result: CompletionResult) -> str:
    return json.dumps(
        {
            "text": result.text,
            "parsed_json": result.parsed.model_dump_json() if result.parsed is not None else None,
            "usage": result.usage.model_dump(),
            "provider": result.provider,
            "model": result.model,
            "tier": result.tier.value,
            "estimated_cost_usd": result.estimated_cost_usd,
            "latency_ms": result.latency_ms,
            "retry_count": result.retry_count,
        }
    )


def _deserialize(
    raw: str, request: CompletionRequest, cache_layer: str
) -> CompletionResult:
    data = json.loads(raw)
    parsed = None
    if data["parsed_json"] is not None and request.response_model is not None:
        parsed = request.response_model.model_validate_json(data["parsed_json"])
    return CompletionResult(
        text=data["text"],
        parsed=parsed,
        usage=Usage(**data["usage"]),
        provider=data["provider"],
        model=data["model"],
        tier=Tier(data["tier"]),
        cache_hit=True,
        cache_layer=cache_layer,  # type: ignore[arg-type]
        estimated_cost_usd=data["estimated_cost_usd"],
        latency_ms=data["latency_ms"],
        retry_count=data["retry_count"],
    )


class ResponseCache:
    def __init__(
        self, redis: Redis, settings: GatewaySettings, embedder: Embedder | None
    ) -> None:
        self._redis = redis
        self._settings = settings
        self._embedder = embedder

    async def get(self, request: CompletionRequest) -> CompletionResult | None:
        key = _cache_key(request)

        exact_raw = await self._redis.get(f"{_EXACT_PREFIX}:{key}")
        if exact_raw is not None:
            return _deserialize(exact_raw, request, "exact")

        if self._embedder is None:
            return None

        query_text = request.messages[-1].content
        query_vector = await self._embedder.embed_query(query_text)

        candidates = await self._redis.lrange(
            f"{_SEMANTIC_PREFIX}:{request.tier.value}", 0, -1
        )
        best_distance = float("inf")
        best_exact_key: str | None = None
        for raw_candidate in candidates:
            candidate = json.loads(raw_candidate)
            distance = _cosine_distance(query_vector, candidate["embedding"])
            if distance < best_distance:
                best_distance = distance
                best_exact_key = candidate["exact_key"]

        threshold = self._settings.cache_semantic_distance_threshold
        if best_exact_key is None or best_distance > threshold:
            return None

        semantic_raw = await self._redis.get(f"{_EXACT_PREFIX}:{best_exact_key}")
        if semantic_raw is None:
            return None
        return _deserialize(semantic_raw, request, "semantic")

    async def put(self, request: CompletionRequest, result: CompletionResult) -> None:
        key = _cache_key(request)
        await self._redis.set(
            f"{_EXACT_PREFIX}:{key}",
            _serialize(result),
            ex=self._settings.cache_exact_ttl_seconds,
        )

        if self._embedder is None:
            return

        query_text = request.messages[-1].content
        query_vector = await self._embedder.embed_query(query_text)
        semantic_key = f"{_SEMANTIC_PREFIX}:{request.tier.value}"
        await self._redis.lpush(
            semantic_key, json.dumps({"embedding": query_vector, "exact_key": key})
        )
        await self._redis.ltrim(semantic_key, 0, self._settings.cache_semantic_max_candidates - 1)
        await self._redis.expire(semantic_key, self._settings.cache_semantic_ttl_seconds)
