---
paths:
  - "packages/llm/**/*.py"
  - "packages/evals/**/*.py"
---

# LLM gateway rules

This project runs on free inference tiers. The gateway is what makes that survivable.

## Routing

Use **LiteLLM** as the router. Do not hand-roll provider dispatch — LiteLLM already
normalises the OpenAI-compatible surface across every provider below, and handles
fallback chains, retries and cost accounting natively. Read its current docs before
adding a provider; the config schema changes.

Model tiers, defined in `packages/llm/registry.py`:

| Tier | Use for | Provider notes |
|---|---|---|
| `fast` | routing, classification, query rewrite | lowest latency free tier; short prompts only |
| `reason` | planning, grounded generation, critic | largest free context window available |
| `bulk` | eval runs, ingestion enrichment, batch | highest daily token quota |
| `local` | offline dev, CI | Ollama; no network, no quota |

Never hardcode a model string outside the registry. Nodes ask for a tier, not a model.

## Fallback chain

Every tier declares an ordered fallback list. A 429 or 5xx moves to the next provider.
Exhausting the chain raises `AllProvidersExhausted`, which the API surfaces as a 503 with
a retry hint — never a 500, and never a silent empty answer.

Free-tier rate limits change roughly monthly and are frequently cut without notice.
Treat every published limit as a moving target: read the ceiling from config, not from a
constant in code, and make the config easy to update.

## Required behaviour

- **Retries:** `tenacity`, exponential backoff with jitter. Respect `Retry-After` when present.
- **Concurrency:** a semaphore per provider, sized below its RPM ceiling. Queuing does not
  fix a rate limit — spreading load across providers does. Prefer failing over to fanning out.
- **Timeouts:** per-call timeout, always set. No unbounded awaits.
- **Budget guard:** per-request and per-day token ceilings. Raise `BudgetExceeded` *before*
  spending, not after.
- **Semantic cache:** Redis, embedding-similarity keyed, with an exact-match layer in front.
  Cache hit rate is a tracked metric and goes in the README.
- **Structured output:** Pydantic v2 schemas. Never parse free text with regex.
- **Telemetry:** every call emits an OTel span with provider, model, tier, tokens in/out,
  estimated cost, cache hit, latency, and retry count.

## Data handling

Free tiers are generally funded by your prompts — assume anything sent to a no-credit-card
free tier may be used for training. This is acceptable here because the corpus is public.
It is not acceptable for user-uploaded content.

- Route anything a user uploads to the `local` tier only, or refuse it.
- State the free-tier data policy plainly in the README and in the UI.
- Free tiers also carry terms limiting high-volume commercial use. This is a portfolio
  demo, which is fine — but never build a client's production system on them.

## Evals

Eval runs dominate token spend. Route them to `bulk`, cache judge calls by
(prompt hash, model, temperature), and sample rather than running the full set on every PR.
Pin the judge model and record it in the scorecard — changing judges invalidates the baseline.
