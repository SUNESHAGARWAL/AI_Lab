#!/usr/bin/env python3
"""Manual smoke test for the LLM gateway's real provider wiring.

Not a pytest test — no fakes, no mocks. Makes one real call through `Gateway.complete`
on each of the fast/reason/bulk tiers using whatever free-tier API keys are in `.env`
(see `.env.example`), and prints provider, latency, tokens, cost, and cache status for
each. Run this yourself after changing `packages/llm/src/llm/registry.py` or rotating
API keys, before building anything on top of the gateway.

Needs a reachable Redis (budget + cache state) — `docker compose up -d` first.

Usage:
    uv run python scripts/smoke_gateway.py
    uv run python scripts/smoke_gateway.py --include-local
    uv run python scripts/smoke_gateway.py --repeat 2   # second call should cache-hit
"""

import argparse
import asyncio

from dotenv import load_dotenv
from llm.errors import AllProvidersExhausted, BudgetExceeded

from llm import CompletionRequest, Gateway, Message, Tier

DEFAULT_TIERS = [Tier.FAST, Tier.REASON, Tier.BULK]


async def _run_tier(gateway: Gateway, tier: Tier) -> bool:
    request = CompletionRequest(
        tier=tier,
        messages=[Message(role="user", content="Reply with exactly one word: pong")],
        max_tokens=16,
    )
    try:
        result = await gateway.complete(request)
    except BudgetExceeded as exc:
        print(f"{tier.value:8} BUDGET EXCEEDED   {exc}")
        return False
    except AllProvidersExhausted as exc:
        print(f"{tier.value:8} ALL EXHAUSTED     {exc}")
        return False

    cache_status = result.cache_layer if result.cache_hit else "miss"
    print(
        f"{tier.value:8} "
        f"provider={result.provider:8} "
        f"model={result.model:32} "
        f"latency={result.latency_ms:7.1f}ms "
        f"tokens_in={result.usage.prompt_tokens:4} "
        f"tokens_out={result.usage.completion_tokens:4} "
        f"cost=${result.estimated_cost_usd:.6f} "
        f"cache={cache_status:8} "
        f"retries={result.retry_count} "
        f"text={result.text.strip()!r}"
    )
    return True


async def main(include_local: bool, repeat: int) -> int:
    load_dotenv()
    gateway = Gateway()

    tiers = [*DEFAULT_TIERS, Tier.LOCAL] if include_local else DEFAULT_TIERS
    print(f"Testing tiers: {', '.join(t.value for t in tiers)}\n")

    ok = True
    for round_number in range(1, repeat + 1):
        if repeat > 1:
            print(f"-- round {round_number} --")
        for tier in tiers:
            ok = await _run_tier(gateway, tier) and ok
        print()

    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="also test the local/Ollama tier (needs a running Ollama daemon)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the whole tier sweep this many times (>1 to observe cache hits)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.include_local, args.repeat)))
