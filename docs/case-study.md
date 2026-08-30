# Case study — AI Lab: a RAG system built to refuse

*Personal project. Source code and live demo are public.*

**Live:** https://ai-lab-web-ten.vercel.app · **Source:** https://github.com/SUNESHAGARWAL/AI_Lab

> INTERNAL — this block is stripped from the PDF build, it isn't shown to clients.
> Use this three ways: attach it as the gallery PDF on the flagship portfolio item, link it
> in proposals as "here's a write-up of how I approach this", and mine it for paragraphs when
> a client asks a specific technical question. It's written for a technical buyer — someone
> who has been burned by a RAG pilot and wants to know whether you'll be different.

---

## The brief I set myself

Build a question-answering system over the EU AI Act and GDPR where **every claim traces back
to a specific article or recital — or the system admits it can't answer.**

The constraint that made it interesting: a confident wrong answer about a regulation is worse
than no answer, because someone acts on it. That inverts the usual RAG success criterion.
The target was never "answer more questions." It was: *when the retrieved text doesn't support
a confident answer, say so.*

Second constraint: **$0/month**, end to end. Not as a stunt — as a design pressure. Free-tier
rate limits are tight and change without warning, and building against them forces the routing,
retry, caching and budget discipline that a paid API lets you skip until it's expensive.

---

## Architecture

Six LangGraph nodes over a single typed `AgentState`. Nodes are pure functions of state and
never call each other — the runtime routes.

```
query → planner → retriever → reranker → generator → critic → hitl_gate → end
             ↓                              ↑            │
          abstain ─────────────────────────┴────────────┘
                          (retry goes back to RETRIEVAL)
```

- **planner** — rewrites the question into a retrieval query, classifies intent, and can
  abstain *before spending a retrieval call* on an out-of-scope question.
- **retriever** — cosine search over pgvector with an HNSW index. 384-dim embeddings from
  `BAAI/bge-small-en-v1.5`, computed **locally on CPU**, never a hosted embedding API.
- **reranker** — currently an identity passthrough. Why, below; it's the most interesting
  decision in the project.
- **generator** — answers from *only* the supplied chunks, emitting structured output with a
  `citations` list and an `abstained` flag. Citations are validated against the real chunk IDs
  in scope; invented IDs are stripped rather than trusted.
- **critic** — reviews the draft for unsupported claims. **Its retry edge goes back to
  retrieval, not to regeneration.** If an answer was thin, better chunks are more likely to fix
  it than another sampling roll. The loop carries a hard iteration cap in state *and* a token
  budget guard enforced by the gateway — the graph decides whether to loop, the gateway decides
  whether the loop is still affordable.
- **hitl_gate** — pauses via LangGraph's `interrupt()` when confidence is low. No bespoke
  pause mechanism; the checkpointer already provides one.

The checkpointer is Postgres-backed, so a thread survives a process restart — which matters
specifically because free-tier compute is ephemeral. A request that outlives a restart resumes
instead of dying.

### The corpus

**1,437 chunks**, indexed at article *and* paragraph grain:

| Source | Articles | Recitals | Chunks |
|---|---:|---:|---:|
| EU AI Act | 113 | 180 | 793 |
| GDPR | 99 | 173 | 644 |

That is every article and every recital of both regulations. It is **not** the AI Act's 13
annexes — see [What it doesn't do](#what-it-doesnt-do).

Ingest is idempotent (`ON CONFLICT DO UPDATE`), so re-running after a source change is safe.

### One gateway, one place that names a model

Every model call in the codebase goes through `packages/llm/gateway.py`. Nothing else imports
a provider SDK. The gateway owns provider routing and fallback chains, retries with backoff,
per-provider concurrency limits, a semantic cache, budget enforcement and telemetry.

That single choke point is the difference between a bad afternoon and a bad week — see the
rate-limit story below.

### Streaming contract

The API streams typed node-level events over SSE (`graph_started`, `node_started`,
`node_completed`, `graph_completed`). The TypeScript event schema is **generated from the
backend's Pydantic models**, so the frontend's live graph visualisation can't silently drift
from the backend's contract. Each node reports its own latency and cost as it completes.

---

## What it measures

Generation, all 37 golden items:

| Metric | Score |
|---|---|
| Appropriate abstention | **36/37** — 1 false abstention, **0 false answers** |
| Faithfulness | **0.9877** |
| Answer relevancy | 0.9767 |
| Citation validity | 0.9054 |
| Context precision | 0.1676 ⚠️ (see below) |

Retrieval, deterministic, 31 scored items (6 out-of-scope excluded):

| k | recall@k | MRR@k | nDCG@k |
|---:|---:|---:|---:|
| 1 | 0.6290 | 0.6452 | 0.6452 |
| 3 | 0.9355 | 0.7634 | 0.8072 |
| 5 | 0.9677 | 0.7699 | 0.8197 |
| 10 | 1.0000 | 0.7735 | 0.8294 |

**Two of these numbers are worse than they used to be, on purpose.**

Recall@5 read a perfect 1.0000 until a question the system genuinely gets wrong was added to
the golden set. A test set that only contains questions you already pass measures nothing, so
the lower number is the more useful one.

**And about that 0.1676.** It's the one metric that looks broken, and it's the one I'd want to
be asked about in an interview. It's an exact set intersection between the golden set's
`relevant_chunk_ids` and the chunks actually shown to the generator. Because the corpus is
chunked at *both* article and paragraph grain, the retriever legitimately returns `article:35`
*and* `article:35:paragraph:7` when the golden set names only one — scored as a miss even
though the generator saw exactly the right text. It's a measurement artifact, it's diagnosed in
an ADR, and it's left on the scorecard rather than quietly dropped.

**The judge is itself audited.** LLM-as-judge numbers are worthless if nobody checks the judge,
so the harness emits a per-case agreement report — every verdict, its reasoning, and the cited
chunk text side by side — designed to be read by a human before the aggregates are trusted.

---

## Three decisions worth reading

### 1. The reranker that got built, measured, and switched off

A local cross-encoder reranker was a stated requirement of my own design doc. I built it,
unit-tested it, then measured `BAAI/bge-reranker-base` against the golden set.

It made retrieval **worse at every k** — including recall@10, meaning it demoted a correctly
retrieved chunk out of the scored range entirely.

| k | recall dense → reranked | nDCG dense → reranked |
|---:|---|---|
| 1 | 0.6290 → 0.5323 | 0.6452 → 0.5484 |
| 3 | 0.9355 → 0.9355 | 0.8072 → 0.7757 |
| 5 | 0.9677 → 0.9355 | 0.8197 → 0.7757 |
| 10 | 1.0000 → 0.9355 | 0.8294 → 0.7757 |

So it ships **disabled**, behind an identity passthrough, with the adapter and the eval flag
kept in the codebase and the evidence written into an ADR.

Building the component was easy. Proving it should be *off* took the eval harness — and the
harness is the reason a plausible, well-tested, requirement-satisfying component didn't quietly
degrade the product.

The working hypothesis, offered as reasoning rather than established fact: a general-domain
reranker has no edge on formally structured legal text, and the cross-encoder's 512-token
window truncates long article-level chunks before scoring them.

### 2. A rate limit that wasn't a rate limit

Eval runs kept dying against Groq, even after waiting a full day for what looked like a daily
quota reset. Reproducing the call *outside* the gateway surfaced the real error body: a
**tokens-per-minute** ceiling — a rolling 60-second window that waiting 24 hours does exactly
nothing for.

DeepSeek became the primary provider, Groq the fallback. The fix was a one-line registry
change *because* all routing lives behind the gateway. If provider calls had been scattered
through the codebase, the same fix would have been a refactor.

The transferable lesson, and the reason it's in this case study: **the failure was in the error
handling, not the provider.** The gateway was swallowing the response body that said exactly
what was wrong.

### 3. `import ragas` doesn't

The eval library I'd planned to use crashes on import against the current
`langchain-community`. I verified it against the current release, confirmed it as a known open
bug rather than my own misconfiguration, switched the judge to `deepeval`, and kept `ragas`
pinned-but-unused with the reason recorded.

Small decision. It's here because "check whether the library actually works before building on
it" is a habit that costs an hour and saves a week.

---

## The part worth reading twice

The golden-set case that justified the whole harness.

Two Layer-3 runs, **identical retrieval**, differing only in the generator's system prompt:

| Run | Abstained? | Faithfulness | Citation validity | Context precision |
|---|---|---|---|---|
| Before prompt hardening | no | **1.0** | **1.0** | **0.0** |
| After | yes | — | 0.0 | **0.0** |

`context_precision = 0.0` in *both* runs. The paragraph that answers the question — GDPR
Article 37(1), the one that lists who must appoint a Data Protection Officer — was **never
among the five chunks the model was shown**.

So in the first run, the model answered "who must appoint a DPO" from the *neighbouring*
paragraphs: group appointments, qualifications, staff and service contracts. And it scored a
**flawless 1.0 on faithfulness, relevancy and citation validity.** Every headline metric
certified the answer as perfect. It was perfectly faithful to text that doesn't answer the
question.

The only number that dissented was the metric flagged above as a measurement artifact. On
average it *is* an artifact of dual-grain chunking. But a **0.0 on a single item** isn't grain —
it means nothing relevant was retrieved at all, and the mean hides that completely.

After hardening, the model abstains — which the abstention metric scores as a *failure*, making
the headline scorecard look worse. **That trade is what the project exists to make.**

### The diagnosis

The refusal is a *ranking* failure, not a corpus gap. Article 37(1) is indexed. Cosine
similarity ranks it **9th** for that query, because the paragraphs of Articles 37 and 38 are so
topically similar that the entire top-10 sits inside a **0.026 score band** — and `bge-small`
can't separate them. The top-5 cut then drops the one paragraph that answers the question.

Re-running those same candidates through the *disabled* cross-encoder moves Article 37(1) from
rank 9 to **rank 2** — comfortably inside the cutoff. The component that fixes this query is
the one the golden set says to leave off.

The trigger is narrower than it looks. The golden set already held the same question in the
statute's own vocabulary — *"when am I required to **designate** a DPO"* — which retrieves at
rank 2. Dropping the article number changes nothing; swapping the verb to **appoint** is what
costs four ranks. And it isn't a general vocabulary gap: everyday paraphrases of other golden
items ("privacy impact assessment", "data leak") still rank 2nd and 1st. The failure needs
*both* an everyday synonym *and* a dense cluster of near-identical siblings competing for the
same five slots.

So rather than flip a switch on the strength of one anecdote, the case went into the golden set
as a permanent failing item, both configurations were re-run over all 37 items, and the reranker
stayed off — because it fixes this one query and breaks more than it fixes.

**That case is left failing on purpose.** It marks the boundary of what this retrieval stack
handles, somewhere it can't be quietly forgotten.

---

## Running on $0

Cost discipline as a design constraint, not a postscript:

- **Embeddings and reranking run on CPU, locally.** Never a hosted embedding API.
- **The demo's headline path costs nothing.** Example questions replay a cached golden trace
  client-side — they work even if the backend is paused, which also means a visitor clicking
  around can't burn the budget.
- **Live queries are rate-limited per IP** and sit behind a global budget guard, so exhaustion
  degrades into a friendly in-band message instead of a 500.
- **The production image went from 8.75 GB to 2.58 GB** after switching to CPU-only torch. The
  default wheel drags in several GB of CUDA libraries that a `device="cpu"` deployment never
  loads.
- Deployed on Railway (API), Vercel (web), Neon (Postgres + pgvector), Upstash (Redis).

---

## Engineering practices

- `packages/core` defines `Protocol` classes and imports no adapters. Dependency arrows point
  inward, always.
- Pydantic v2 models cross every module boundary. No bare dicts between packages, no regex
  parsing of model output — structured output only.
- Every graph node ships with three things: a unit test, a golden-set eval case, and an
  OpenTelemetry span. A node without all three isn't done.
- 251 fast deterministic tests, runnable without any model call.
- Retrieved documents are treated as **untrusted input**: an instruction found inside a
  retrieved chunk is data to display, never a command to follow. The public demo has a rate
  limit, a length cap and an input guard on every path.
- Six ADRs record the decisions, including the ones that reversed earlier decisions.
- Config failures are loud at startup. A missing credential is never silently defaulted.

---

## What it doesn't do

Stated plainly, because a case study that only lists wins isn't information:

- **Token-by-token answer streaming.** The contract streams node-level events; the answer
  arrives as one event on completion. True token streaming needs its own decision record
  covering the interactions with the gateway's semantic cache (cached responses have no
  stream), retry logic (partial-then-retry), and the budget guard (mid-stream cutoff).
  Deferred deliberately rather than half-built.
- **Layer 4 of the eval harness** (adversarial and robustness testing) isn't built.
- **The `appoint`-vocabulary retrieval failure** described above is open. The fix is probably a
  hybrid BM25 + dense retriever or a domain-tuned reranker — both are real work, and neither is
  justified by a single failing item without measuring first.
- **Two regulations, English only.** Nothing in the pipeline is specific to them, but that's
  what's been ingested and evaluated.
- **The EU AI Act's annexes aren't ingested.** Every article and recital of both regulations
  is (113/113 and 180/180 for the AI Act, 99/99 and 173/173 for GDPR), but none of the Act's
  13 annexes. That matters more than it sounds: Annex III is the list of high-risk AI systems
  that Article 6(2) defines "high-risk" by pointing at, so "is my recruitment tool high-risk?"
  is a question this corpus cannot fully answer. The parser extracts article and recital
  subdivisions only. Asked to list Annex III, the system abstains and names the reason — the
  design working as intended, which is not the same as the gap being fine.

---

## What this would look like on your project

The parts that transfer directly:

1. **Evals before features.** A golden set with provenance, built early, so every subsequent
   change has a before-and-after instead of an opinion.
2. **One gateway for model calls.** Provider switches become config changes. Budget and
   rate-limit handling live in one auditable place.
3. **Abstention as a first-class outcome.** If a wrong answer costs you something, the system
   needs to be able to decline — and you need a metric that tells you whether it declines at
   the right times.
4. **Decisions written down.** Six months later, "why is the reranker off?" has an answer with
   numbers attached instead of a shrug.

If a wrong answer in your domain is expensive, that's the conversation I'd want to start with.
