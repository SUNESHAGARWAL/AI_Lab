# 0001 — Orchestration: LangGraph + AI SDK v6, model calls behind a LiteLLM gateway

## Status

Accepted — 2026-08-05

## Context

The backend runs a multi-step agentic RAG pipeline: query rewrite → retrieve → rerank →
generate → critique/self-correct → cite-check → (optional) abstain. This needs explicit,
inspectable control flow with retry loops that have hard caps, human-in-the-loop pauses for
review, and state that survives a process restart — this is a free-tier deployment on
ephemeral compute, so a request that outlives a dyno restart must resume, not die.

On the frontend, the chat UI needs token streaming, incremental citation rendering as the
answer is produced, and a way to surface a human-approval gate (e.g., "this query is
ambiguous, did you mean X or Y?") without inventing a bespoke SSE protocol.

All model calls must be provider-agnostic and centrally governed for cost, because the
project runs on $0 and free-tier rate limits are tight and change monthly per
`CLAUDE.md`.

Versions verified against current docs/changelogs as of 2026-08-05:

- **LangGraph** (Python): core package `langgraph==1.2.10` (PyPI, released 2026-07-28).
  `langgraph-checkpoint-postgres==3.1.0` (2026-05-12) provides the Postgres-backed
  checkpointer. `interrupt()` is the current primitive for human-in-the-loop pauses,
  replacing the older `interrupt_before`/`interrupt_after` compile-time flags for most
  new graphs — those flags still work but are considered lower-level.
- **Vercel AI SDK**: `ai@6.0.x` (npm, e.g. `6.0.204`). v6 is a major breaking release
  that ships a new v3 Language Model Specification, a first-class `Agent` /
  `ToolLoopAgent` abstraction, human-in-the-loop tool approval, stable structured
  outputs with tool calling, a new streaming wire format, and DevTools. Migration from
  v5 has a codemod (`npx @ai-sdk/codemod v6`).
- **LiteLLM**: `litellm==1.95.0` (PyPI, 2026-08-02) as the Python SDK embedded inside
  our own gateway module (not run as a standalone proxy process, to keep infra at zero
  services). Router-level fallback and cooldown behavior: on a 429 the failing
  deployment is placed on cooldown and the router retries against backup deployments;
  `token_bucket` retry policy is now recommended over plain exponential backoff to avoid
  request pile-ups under tight free-tier RPM/TPM limits.

## Decision

1. **LangGraph is the orchestration runtime for `apps/api`.** Graph nodes are pure
   functions over a single typed `AgentState` TypedDict, per the graph conventions in
   `CLAUDE.md`. The runtime routes between nodes; nodes never call each other directly.
2. **Postgres is the only checkpointer**, via `langgraph-checkpoint-postgres`, in every
   environment except unit tests (which may use the in-memory saver). This is what
   makes free-tier, restart-prone hosting viable — a thread can be paused at an
   `interrupt()` and resumed hours later without losing state.
3. **`interrupt()` is the only human-in-the-loop mechanism.** No custom pause/resume
   protocol, no polling flags, no side-channel queue — this matches the explicit
   instruction in `CLAUDE.md` and avoids reinventing what the checkpointer already gives
   us for free.
4. **Every loop (critic retries, query rewrites) gets a hard iteration cap carried in
   `AgentState` plus a token-budget guard enforced by the gateway**, not by the graph.
   The graph decides *whether* to loop; the gateway decides whether the loop is still
   affordable.
5. **The Next.js frontend uses AI SDK v6's `Agent`/`ToolLoopAgent` + `useChat` message-
   parts model** to stream tokens and structured citation parts, and to render the
   `interrupt()` pause as a tool-approval-style UI gate rather than a custom modal.
   The LangGraph backend emits an SSE/data-stream response shaped to the AI SDK v6
   wire format so the two sides don't need a translation layer maintained by us.
6. **No direct provider SDK calls anywhere.** Every model call — inside a LangGraph
   node, in the ingest CLI, in eval runners — goes through `packages/llm/gateway.py`,
   which wraps LiteLLM's Python SDK (embedded, not the standalone proxy — one fewer
   moving part to host for free). The gateway owns:
   - provider routing and fallback chains (free-tier daily quota exhausted → next free
     provider, not a paid one, without asking first),
   - retry with backoff, switched to LiteLLM's `token_bucket` retry policy rather than
     naive exponential backoff, since free-tier 429s are frequent and expected,
   - response caching (critical for eval runs, which are the biggest token consumer),
   - budget enforcement (hard stop, not a warning, when a request would exceed the
     configured budget), and
   - OpenTelemetry spans + token-count/hash-only logging (never full prompt/response
     bodies at INFO, per the cost/security rules).

## Alternatives considered

- **CrewAI / AutoGen for orchestration.** Rejected: both impose their own agent-loop
  abstraction on top of the model, which conflicts with "don't build a new abstraction
  over LangGraph — use its primitives" and, in AutoGen's case, its own primitives. We'd
  be adopting a second orchestration layer for no capability LangGraph lacks.
  LangGraph's explicit graph + typed state also makes the eval/telemetry requirement
  (a span and a golden-set case per node) straightforward — a node is a natural unit of
  instrumentation; an implicit agent loop is not.
- **Custom asyncio state machine instead of LangGraph.** Rejected: we'd end up
  rebuilding checkpointing, interrupt/resume, and time-travel debugging from scratch.
  LangGraph's Postgres checkpointer is exactly the "survive a restart on free-tier
  hosting" property we need, and building it ourselves is the kind of utility
  `CLAUDE.md` says must be justified against an existing library, not written.
- **Vercel AI SDK v5 (stay on the older major).** Rejected: v5 lacks the `Agent`/
  `ToolLoopAgent` abstraction and native tool-approval flow, which is exactly the UI
  primitive we need for `interrupt()`-driven human-in-the-loop gates. Adopting v5 now
  means a forced migration later; the codemod makes moving to v6 upfront cheap enough
  that "verify APIs against current docs, don't guess a signature" tips the decision
  toward the current major.
- **Raw provider SDKs (openai-python, anthropic-sdk) directly in graph nodes.**
  Rejected outright by the non-negotiables: no direct provider SDK calls outside the
  gateway. This would also make free-tier fallback (provider A quota exhausted →
  provider B) a per-call-site concern instead of a single gateway policy.
- **LiteLLM Proxy (standalone server) instead of the embedded Python SDK.** Rejected
  for now: it's an extra service to host and keep up on a $0 infra budget (`just dev`
  already runs api, web, postgres, redis). The embedded SDK gives us the same routing/
  fallback/retry logic inside `packages/llm/gateway.py` without adding a process. If the
  team later runs multiple services needing shared rate-limit state, revisit — the
  proxy's centralized cooldown tracking is the thing we'd be giving up.
- **LangGraph's `interrupt_before`/`interrupt_after` compile-time flags instead of
  `interrupt()`.** Rejected as the primary mechanism: they pause *at* a node boundary
  chosen at compile time, whereas `interrupt()` pauses *inside* a node at the exact
  point a human decision is needed (e.g., mid-way through query disambiguation), which
  fits our abstention/clarification flows better. We may still use the compile-time
  flags for coarse debugging breakpoints in dev.

## Trade-offs

- LangGraph's node-level explicitness (no direct node-to-node calls) means more
  boilerplate routing logic than an implicit agent loop would need — accepted, because
  the eval-per-node and span-per-node requirements make that explicitness pay for
  itself in testability.
- Embedding LiteLLM as a library instead of running its proxy means gateway state
  (cooldowns, budget counters) lives in-process per API instance rather than shared
  centrally — acceptable at current scale (single API service), but will need Redis-
  backed shared state if we ever run multiple API replicas. Redis is already in
  `just dev`, so this is a future gateway change, not a new dependency.
- AI SDK v6 is a large breaking release; pinning it now means absorbing its migration
  cost immediately rather than deferring it, but deferring would mean building the
  tool-approval UI against v5 and re-doing it later — worse.
- Postgres-backed checkpointing adds write load to the same Postgres instance that
  holds pgvector data and app tables (see 0002). At $0-infra scale this is a non-issue;
  it becomes a capacity-planning question only if traffic grows past free-tier Postgres
  limits, at which point checkpoint data (ephemeral, prunable) is the easiest thing to
  move out.

## Consequences

- `packages/llm/gateway.py` is the single required integration point for any new model
  call; adding a call site elsewhere is a code-review blocker.
- Every new LangGraph node PR must include a unit test, a golden-set eval case, and an
  OpenTelemetry span, or it isn't done, per `CLAUDE.md`.
- `pyproject.toml` pins `langgraph==1.2.10`, `langgraph-checkpoint-postgres==3.1.0`,
  `litellm==1.95.0`; `package.json` pins `ai@6.0.204` (or the latest 6.0.x patch at
  merge time — re-verify before pinning, this doc is a snapshot).
- Before implementing, re-check `langgraph-checkpoint-postgres` release notes for the
  "parallel interrupts" changelog entry (multiple interrupt chunks merging) — it's
  recent and relevant to any node that raises more than one `interrupt()`.
