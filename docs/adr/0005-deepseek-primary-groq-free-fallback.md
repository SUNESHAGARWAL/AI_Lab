# 0005 — DeepSeek as primary provider, Groq free tier as fallback

## Status

Accepted — 2026-08-07

## Context

Layer 3 generation-eval runs kept failing with `RateLimitError`/`AllProvidersExhausted`
against Groq, even right after a fresh 24-hour period — contradicting the assumption
that this was the daily quota. Reproduced the failure directly against Groq's API
(bypassing our gateway entirely) to get the real error body, not just LiteLLM's
mapped exception type:

```
"Rate limit reached for model `llama-3.1-8b-instant` in organization ... on tokens
per minute (TPM): Limit 6000, Used 3038, Requested 3080. Please try again in 1.18s."
```

**Root cause: a tokens-per-minute (TPM) limit, not the daily request quota (RPD).**
RPD resets every 24h; TPM is a separate, much tighter, continuously-rolling
60-second window — "24 hours passed" doesn't touch it at all. Checked every Groq
model then in the registry (`packages/llm/src/llm/registry.py`, prior task) against
this account's real limits:

| model | TPM | RPD |
|---|---|---|
| `llama-3.1-8b-instant` | 6,000 | 14,400 |
| `openai/gpt-oss-20b` | 8,000 | 1,000 |
| `llama-3.3-70b-versatile` | 12,000 | 1,000 |
| `openai/gpt-oss-120b` | 8,000 | 1,000 |

All tight against this project's real prompt sizes — golden-set questions retrieve
full AI Act/GDPR article chunks, and a single ~3,000-token test prompt alone used
half of `llama-3.1-8b-instant`'s 6,000 TPM budget. A normal Layer 3 item (generator +
per-citation judge + faithfulness + answer_relevancy) easily exceeds that within a
few seconds, on **both** Groq fallback models at once (same account, shared usage).

**Cost model against real usage.** Estimated ~6,000 tokens/conversation-turn from
this project's actual prompt sizes (planner + generator with up to 5 chunks +
critic, occasional retries). Scenario: 10-20 users/month, 10-30 conversations each,
~3 turns/conversation → 1.8M-10.8M tokens/month. Compared providers at both ends
(blended 75% input / 25% output):

| Provider | Model | $/M in・out | Low (1.8M tok) | High (10.8M tok) |
|---|---|---|---|---|
| Groq (paid Dev Tier) | 8b/70b mix | $0.05-0.59 / $0.08-0.79 | $0.88 | $5.30 |
| **DeepSeek** | deepseek-chat (V4-Flash) | $0.14 / $0.28 | $0.32 | $1.89 |
| Kimi K2.5 | kimi-k2.5 | $0.60 / $3.00 | $2.16 | $12.96 (over budget) |
| Gemini 2.5 Flash-Lite | — | $0.10 / $0.40 | $0.94 | $5.65 |
| HF Inference Providers | routes to Groq/Together/etc. | provider rate + small credit | ~Groq | ~Groq |

DeepSeek's free tier (500K tokens/**day** = 15M/month, no per-minute wall) alone
covers even the high-end estimate for $0; a new-account 5M-token (~$8.40) sign-up
grant covers the first month either way if the daily allowance is ever exceeded on a
single day. Even fully paid, DeepSeek stays far under the $8/month ceiling at both
usage ends. Kimi is the only option that risks exceeding it, from output-heavy
pricing. HF Inference Providers is just a pass-through router to the same
providers (Groq, Together, etc.) at their own rates plus a small monthly credit —
no pricing advantage, adds a hop, not adopted.

**Structured-output bonus, not the deciding factor but real.** Confirmed via
`litellm.supports_response_schema(model="deepseek/deepseek-chat",
custom_llm_provider="deepseek")` → `True`: DeepSeek supports native structured
output. Unlike Groq's small models (confirmed `False` for all four in the table
above), LiteLLM never falls back to a forced tool call for DeepSeek — so routing
through it removes the whole `tool_use_failed` failure class (see
`packages/llm/src/llm/prompted_json.py`, built in the prior task to work around
this specifically for the judge/critic) for *every* structured-output node,
including the planner and generator that task deliberately left unpatched.

## Decision

1. **DeepSeek is the primary provider for every network tier** (`FAST`, `BULK`,
   `REASON`) in `packages/llm/src/llm/registry.py`: `deepseek/deepseek-chat` for
   `FAST`/`BULK`, `deepseek/deepseek-reasoner` for `REASON` (same price; the split
   matches each tier's stated purpose, not cost). Requires `DEEPSEEK_API_KEY` in
   the environment — added to `.env`/`.env.example` alongside the existing
   `GROQ_API_KEY`/`GEMINI_API_KEY` entries, same LiteLLM auto-read convention.
2. **Groq's free tier stays in every chain as the fallback**, unchanged from the
   prior task's entries — it still works, just TPM-constrained, which is strictly
   better than no fallback at all.
3. **Groq's paid Dev Tier is not added yet** — the account isn't set up for it. Per
   explicit instruction, add it (ahead of the free entries) once it is; that would
   remove the TPM ceiling from the fallback path too, not just the primary.
4. **Gemini stays disabled**, commented out (not deleted) in the registry, for two
   independent reasons found during this investigation: its free-tier quota issue
   is still unresolved on this account, and separately its currently-configured
   model ids (`gemini-2.0-flash`, `gemini-2.0-flash-lite`) are deprecated as of
   mid-2026 — a future re-enable must move to `gemini-2.5-flash-lite` or newer
   regardless of the quota question.

## Alternatives considered

- **Just increase Groq's retry attempts/backoff to ride out the TPM window.**
  Would help marginally (the 429 body's own `retry-after` is ~1-3s) but doesn't
  address that this workload's real token volume routinely exceeds a 6,000-12,000
  TPM ceiling within normal (non-adversarial) operation — a structural mismatch,
  not a transient blip to retry through.
- **Truncate chunk text sent to judges/generator to fit under Groq's TPM.** Reduces
  the citation/faithfulness judge's evidence, directly at odds with citation
  validity being one of this project's core eval metrics — rejected as fixing the
  symptom by degrading the thing being measured.
- **Kimi K2.5 as primary.** Rejected on cost: exceeds the $8/month ceiling at the
  scenario's high end from output-token pricing ($3.00/M), the most expensive of
  the four compared.
- **Gemini, once its quota issue is resolved.** Not rejected, deferred — worth
  revisiting on `gemini-2.5-flash-lite` as an additional fallback (or reinstated
  demo-tier primary per the original ADR 0001 intent) once account access is
  confirmed working; no reason to run only two providers if a third free option is
  genuinely available.

## Trade-offs

- One more provider (`deepseek`) in the gateway's dependency surface — accepted,
  same shape as any other entry in `TierRegistry`, no new abstraction.
- DeepSeek is a Chinese-domiciled provider; its data-handling terms differ from
  Groq's/Gemini's. `.claude/rules/llm-gateway.md`'s existing "free tiers are
  generally funded by your prompts" data-policy note already covers this in kind —
  no different in principle from the Groq/Gemini assumption already accepted for
  this public, non-sensitive corpus (EU AI Act/GDPR text). Re-verify DeepSeek's
  specific terms before this project handles anything beyond that public corpus.
- `GatewaySettings.gemini_daily_request_ceiling`/`gemini_soft_limit_fraction` and
  `build_default_registry`'s Gemini-specific ceiling branch are now fully inert (no
  Gemini entries exist to apply them to) — left in place rather than removed, since
  Gemini re-enabling is a near-term, not hypothetical, follow-up.

## Consequences

- Real live calls require a real `DEEPSEEK_API_KEY` in `.env` — added the env var
  slot, not a value; the gateway will raise a clear provider-auth error on first
  use if it's still empty, not silently no-op or fall through unnoticed (Groq's
  fallback entries would still work in that case, just without DeepSeek's benefit).
- Future reranker/model-choice ADRs (0002, 0003) and the judge-routing ADR (0004)
  are unaffected — this ADR only changes which provider serves each `Tier`, not the
  gateway's structured-output mechanism, retry/cache/budget behavior, or the
  reranker decision.
- Next natural follow-ups, in rough priority: (a) add Groq's paid Dev Tier as a
  fallback once available, ahead of the free entries; (b) re-verify DeepSeek's
  actual real-world TPM/RPD ceilings against this account (the 500K/day figure is
  from published documentation, not yet confirmed against this project's own usage
  the way Groq's was); (c) revisit Gemini on current model ids once its account
  issue is resolved.
