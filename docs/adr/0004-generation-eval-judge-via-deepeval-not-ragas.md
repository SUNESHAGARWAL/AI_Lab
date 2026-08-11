# 0004 — Layer 3 generation eval: judge via deepeval (not ragas), and evals→api

## Status

Accepted — 2026-08-06

## Context

`CLAUDE.md`/the plan's Layer 3 named `ragas` for the standard generation-quality
metrics (faithfulness, answer relevancy, context precision), already pinned in
`packages/evals/pyproject.toml` alongside `deepeval`. Verifying its current API
before writing any code (per `CLAUDE.md`'s "verify APIs against current docs before
use" rule) surfaced two blockers, not one:

**`import ragas` fails today.** `ragas==0.4.3` — the latest release on PyPI as of
this writing — crashes on import:
```
File ".../ragas/llms/base.py", line 12, in <module>
    from langchain_community.chat_models.vertexai import ChatVertexAI
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```
A GitHub issue search on `explodinggradients/ragas` confirms this is a known,
currently open packaging bug: `langchain_community.chat_models.vertexai` was removed
from newer `langchain-community` releases and ragas's own import chain hasn't caught
up (related PRs #2793/#2810 attempt import-path fixes, unmerged/unreleased as of
`0.4.3`). This isn't a local environment problem — the installed `langchain-community`
(`0.4.2`, ragas's own declared dependency) simply doesn't ship the module ragas
imports at package-load time.

**Even fixed, ragas's custom-LLM interface is a heavy, wrong-layer integration for
this codebase.** `CLAUDE.md`'s non-negotiable: "All model calls go through
`packages/llm/gateway.py`. No direct provider SDK calls anywhere else." Ragas's
default LLM factories call OpenAI directly — using ragas "as shipped" would silently
violate that. Routing ragas's judge calls through our own gateway means implementing
`ragas.llms.base.BaseRagasLLM`: `generate_text`/`agenerate_text` returning
langchain's `LLMResult` over a `PromptValue`, plus a `Callbacks` parameter — real
coupling to langchain's internal types, for the sole purpose of getting the same
three canned metric prompts a much lighter integration could get us.

`deepeval==4.1.5` (already pinned) has the same three metrics
(`FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`) and
imports cleanly. Its custom-model extension point, `DeepEvalBaseLLM`, needs exactly
four methods — `load_model`, `generate`, `a_generate`, `get_model_name` — all plain
strings in, plain strings out, no langchain types anywhere.

**`ContextualPrecisionMetric` doesn't fit this project's golden-set shape either.**
It requires an `expected_output` (a reference answer) to judge which retrieved
context nodes are actually relevant. `evals/golden.py`'s `GoldenItem` schema has no
model-answer field — only human-reviewed `relevant_chunk_ids` (see
`evals/candidates.py`'s interactive review tool, which is exactly how those ids got
verified). Since real ground truth already exists for precisely the question this
metric answers, an LLM judge against a synthetic `expected_output` would be strictly
weaker signal than computing it directly.

**Running "the full graph end-to-end" requires `packages/evals` to import
`apps/api`.** `build_graph`, `AgentState`, `initial_state`, and every node factory
live in `apps/api/src/api/graph/` — there is no other product surface to invoke them
through (`apps/api` exposes only `/health`; no chat/query endpoint exists).
`evals/golden.py`'s own comment already flags "packages/evals must not depend on
apps/api — wrong dependency direction" for the unrelated `QuestionType`/`QueryIntent`
taxonomy question; this task needs the actual graph, which that comment's stated
rule would forbid importing at all.

## Decision

1. **Layer 3's "standard" metrics (faithfulness, answer relevancy) use `deepeval`**,
   judged through a new `evals.judge.GatewayJudgeModel(DeepEvalBaseLLM)` that routes
   every call through `llm.Gateway.complete()` — inheriting retries, the budget
   guard, exact-match response caching, and OTel telemetry for free, and keeping
   every model call in the codebase going through the one required entry point.
2. **`ragas` stays pinned in `packages/evals/pyproject.toml`, unused**, with a
   comment pointing at this ADR. Revisit if a future ragas release fixes the import
   *and* the value of switching (a slightly different metric implementation) is
   judged worth re-wiring the `BaseRagasLLM` adapter — not planned unless something
   changes.
3. **"Context precision" is computed deterministically** from the golden set's own
   `relevant_chunk_ids` against the chunk ids actually shown to the generator
   (`state["reranked_chunks"]`), not via `ContextualPrecisionMetric`. See
   `evals/generation_metrics.py::context_precision`.
4. **`packages/evals` depends on `apps.api` directly** (`packages/evals/pyproject.toml`
   gains an `api` workspace dependency), a deliberate, narrow, one-directional
   exception to the general "packages don't import apps" convention. This is safe
   specifically because `apps/api` is a leaf node in this project's dependency
   graph — nothing (`packages/*` or `apps/web`) imports `apps/api`, so `evals → api`
   introduces no cycle and leaks no app-layer code into anything `apps/api` itself
   depends on. It's the same shape as a test suite importing the application under
   test, not a package-to-package layering violation. `packages/core` staying
   adapter-free (the actual concern `CLAUDE.md`'s "dependency arrows point inward"
   line protects) is untouched by this.

## Alternatives considered

- **Fix ragas's import** (stub-install `langchain-google-vertexai`, or pin an older
  `langchain-community` that still has the removed module) and proceed with
  `BaseRagasLLM`. Rejected: fixes the import but not the deeper architectural
  mismatch (langchain-coupled custom-LLM interface); `deepeval` gets the same three
  metrics for a fraction of the integration surface.
- **Extract the graph into a new `packages/graph` package** so `packages/evals`
  never has to import an app. Architecturally the "purer" fix for the
  dependency-direction question, but a real refactor of already-tested, working code
  (`schemas.py`, `state.py`, `nodes.py`, `build.py`, all from earlier sessions) that
  this task didn't ask for and doesn't need — deferred until a second real consumer
  of the graph (not just an eval harness) shows up.
- **Build a `/query` HTTP endpoint and have `evals` call it over HTTP** instead of
  importing the graph in-process. Avoids the import-direction question entirely, but
  means building a whole new product surface (routing, request/response schemas,
  auth surface) as a side effect of an eval-harness task — out of scope, and
  premature ahead of the actual API design work.
- **Use `ContextualPrecisionMetric` with a placeholder `expected_output`** (e.g. the
  concatenated relevant chunk texts). Rejected: fabricating a reference answer to
  satisfy a metric's required field, when exact ground truth for the underlying
  question already exists, would produce a noisier number for no benefit.

## Trade-offs

- `packages/evals`' dependency footprint grows to include `apps/api` (and
  transitively `langgraph`, `fastapi`, etc.) — heavier than a pure metrics package,
  accepted because there's no other way to run the real product graph without
  building product surface this task doesn't need.
- Two eval-adjacent frameworks (`ragas`, `deepeval`) are declared as dependencies but
  only one is used. Slight `uv sync` overhead, no runtime cost — acceptable rather
  than removing a dependency that may become usable after a ragas fix.
- `citation_validity`'s "supports the claim" check judges each citation against the
  whole answer text, not a specific claim within it — `GeneratedAnswer.citations` is
  a flat, answer-level list with no per-sentence claim mapping anywhere in this
  codebase. This is the finest grain the actual schema supports, not a shortcut
  taken to save work; a future schema change (per-claim citations) would let this
  metric get more precise without changing its interface.

## Consequences

- If `apps/api`'s graph topology or `AgentState` shape changes, `evals.generation_run`
  needs updating in lockstep — there's no port/interface insulating this package
  from those internals, unlike `packages/evals`' Layer 1/2 dependency on `core.ports`.
  Acceptable for now given point 4 above; would be the first sign it's time to
  reconsider the `packages/graph` extraction noted above.
- Re-evaluating ragas later means writing the `BaseRagasLLM` adapter this ADR chose
  not to build today — not a small follow-up, a genuinely separate task.
- `judge-agreement-report`'s human-validation step (a small, hand-picked sample read
  by a person) remains the gate on trusting `run-generation-eval`'s aggregate numbers
  at scale, same discipline `evals/candidates.py`'s human golden-set review already
  established — this ADR doesn't change that requirement.
