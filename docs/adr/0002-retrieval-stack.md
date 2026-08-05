# 0002 — Retrieval stack: Postgres + pgvector hybrid search, local cross-encoder reranking, dual-track evals

## Status

Accepted — 2026-08-05

## Context

The product's core promise is citation-grounded answers with an honest abstention when
the corpus doesn't support a claim. That makes retrieval quality the dominant factor in
end-to-end quality, and it makes retrieval eval (citation validity, appropriate
abstention) a first-class deliverable, not an afterthought.

Constraints from `CLAUDE.md`:

- Runs on $0 — no hosted vector DB, no hosted embedding/reranking API, ever, without
  asking first.
- Embeddings and reranking run locally on CPU via `sentence-transformers`.
- Postgres is already the system of record (also backs LangGraph's checkpointer per
  0001), so any storage decision competes for the same free-tier instance.
- The golden set (`evals/datasets/`) is the source of truth for quality; changing
  retrieval without re-running `just evals-fast` is not allowed, and eval runs are the
  biggest token consumer, so eval token cost is a retrieval-stack concern too.

Versions/facts verified against current docs as of 2026-08-05:

- **pgvector**: HNSW is now the recommended default index type for approximate nearest-
  neighbor search (better query-time recall/speed trade-off than IVFFlat, no training
  pass required); IVFFlat remains preferable only for small, mostly-static tables where
  build time/memory matter more than recall.
- **Hybrid search pattern**: combine pgvector's cosine/L2 distance operators with
  Postgres native full-text search (`tsvector`/`tsquery`, `ts_rank`) and merge the two
  ranked lists with Reciprocal Rank Fusion (RRF) — this is the documented 2026 pattern
  for "BM25-ish + vector" search inside plain Postgres, with no extra service. Pulling a
  wider candidate set from each side (e.g., top-20 each) before fusing to a smaller
  final set (e.g., top-10) outperforms pulling the final size from each side directly.
- **Reranking**: `sentence-transformers`'s `CrossEncoder` API is the current, documented
  way to do local pairwise reranking. `cross-encoder/ms-marco-MiniLM-L-6-v2` is the
  standard small default — fast and small, trained on MS MARCO passage ranking, with
  reported CPU latency of ~100–250ms for N=50 candidates on a MiniLM cross-encoder,
  which is well inside a synchronous request budget.
- **Eval frameworks**: `deepeval==4.1.5` (PyPI, 2026-07-31) and `ragas` (latest release
  2026-01-13). Both compute faithfulness, answer relevancy, context precision, and
  context recall. DeepEval's differentiators relevant to us: `GEval` for custom
  criteria-based scoring, pytest-native pass/fail gates suited to CI (`just evals-fast`
  running deterministic layers), and `ContextualPrecisionMetric`/
  `ContextualRecallMetric`/`ContextualRelevancyMetric` for retrieval-specific scoring.
  Ragas's differentiator: deeper support for score tables/trend tracking across a
  dataset and additional academic metrics (context entity recall, factual
  correctness) — better suited to the nightly, non-CI-blocking deep pass.

## Decision

1. **Postgres + pgvector is the only vector store**, storing chunk embeddings alongside
   chunk text and metadata in the same database as app state and LangGraph checkpoints.
   No separate vector DB service — that would be a second stateful service to run for
   free and a second place citations could drift out of sync with source documents.
2. **HNSW is the index type**, not IVFFlat. The corpus is expected to grow and be
   re-ingested (`just ingest`) rather than stay small and static, so HNSW's recall/speed
   profile and lack of a training pass win.
3. **Retrieval is hybrid**: pgvector cosine similarity for semantic recall + Postgres
   `tsvector`/`ts_rank` full-text search for exact-term/keyword recall (catches acronyms,
   IDs, quoted phrases that embeddings blur), fused with Reciprocal Rank Fusion. Both
   queries run against the same Postgres instance — no BM25 microservice, no
   Elasticsearch/OpenSearch/ParadeDB. Candidate depth: top-20 from each branch, fused
   down to a smaller top-k before reranking.
4. **Reranking is a local `sentence-transformers` `CrossEncoder`**
   (`cross-encoder/ms-marco-MiniLM-L-6-v2` as the default model, swappable via config),
   run on CPU inside `packages/retrieval`. This is a hard requirement from `CLAUDE.md`,
   not a choice we're weighing against a hosted reranking API.
5. **Both `packages/core` retrieval Protocols and `packages/retrieval` adapters are
   split by concern**: an embedding port, a lexical-search port, a fusion port, and a
   rerank port, each independently swappable (e.g., swapping HNSW params, or the
   cross-encoder checkpoint, without touching the fusion or generation code).
6. **Evals run on both frameworks, at different cadences**: DeepEval metrics
   (`GEval` for answer quality, `ContextualPrecisionMetric`/`ContextualRecallMetric` for
   retrieval) are wired into `just evals-fast` as pytest-native pass/fail gates — no
   judge-model call, deterministic layers only, safe for every PR. Ragas runs in
   `just evals-full` (nightly, judge-model-backed) for broader/trend-tracked scoring
   (faithfulness, context recall/precision, factual correctness) across the full golden
   set. Both frameworks' judge-model calls, when they need one, go through the same
   `packages/llm/gateway.py` as everything else (0001) — routed to whichever free
   provider has quota that day, cached aggressively since eval runs are the biggest
   token consumer.
7. **Citation validity and appropriate abstention are custom metrics** implemented in
   `packages/evals`, not borrowed from either framework's built-ins, since neither
   DeepEval nor Ragas ships a metric that checks "does every cited reference exist in
   the corpus and actually support the claim" or "did the system correctly refuse."
   They're built as DeepEval custom metrics (for the fast/CI gate) with a Ragas-based
   nightly counterpart for trend tracking.

## Alternatives considered

- **Dedicated vector DB (Qdrant, Weaviate, Pinecone, Chroma server).** Rejected: all
  require either a hosted paid tier at meaningful scale or a second self-hosted service
  outside the `just dev` compose stack (api, web, postgres, redis). pgvector with HNSW
  now gets close enough on recall/latency for a QA corpus at this scale that the
  operational simplicity of "one Postgres" wins outright, and it keeps citations
  (source-of-truth text) and their embeddings transactionally consistent in one store.
- **Vector-only (no hybrid/BM25) retrieval.** Rejected: semantic-only search reliably
  misses exact-match queries (error codes, proper nouns, quoted terms) that matter for a
  citation-grounded QA product — a wrong or missing citation is a worse failure mode
  here than in a general chat app, so the extra `tsvector` query and RRF merge step is
  worth the added latency.
- **A standalone BM25 engine (Elasticsearch, OpenSearch, ParadeDB's `pg_bm25`
  extension).** Rejected for the same reason as a dedicated vector DB: extra service or
  extra Postgres extension outside what's guaranteed available on free-tier managed
  Postgres. Native `tsvector`/`ts_rank` is not true BM25, but combined with RRF fusion
  and reranking it recovers most of the practical benefit without the infra cost —
  revisit only if eval numbers show the ranking-function gap actually matters here.
- **Hosted reranking API (Cohere Rerank, Jina Reranker).** Rejected outright by
  `CLAUDE.md`: no hosted embedding/reranking API, ever, without asking first. Local
  `CrossEncoder` reranking is not a trade-off we're weighing — it's a constraint.
- **Skip reranking, rely on RRF-fused hybrid score alone.** Rejected: cross-encoder
  reranking is materially more accurate than any dense/sparse fusion score because it
  jointly attends over the (query, chunk) pair rather than comparing independent
  embeddings, and the CPU latency cost (~100–250ms for N≈50) is affordable inside a
  streaming response budget. Given citation validity is a named eval metric, the
  accuracy gain is directly worth the latency.
- **Ragas only, or DeepEval only, instead of both.** Rejected: DeepEval's pytest-native
  pass/fail gates are what makes `just evals-fast` meaningfully CI-blocking without a
  judge-model call; Ragas's dataset-level trend tracking is what makes `just evals-full`
  useful for catching slow regressions across the golden set. Using only one framework
  means giving up one of those two properties. The cost is maintaining two eval
  dependencies and two metric-definition surfaces — mitigated by keeping the *custom*
  metrics (citation validity, abstention) in our own `packages/evals` code so they don't
  fork across frameworks.

## Trade-offs

- Native Postgres full-text search is a real-but-imperfect stand-in for BM25 — accepted
  as a $0-infra trade, to be revisited only if golden-set eval scores show a keyword-
  recall gap that RRF + reranking doesn't close.
- Running reranking on CPU inside the request path adds latency (tens to a couple
  hundred ms per query) that a hosted GPU reranker wouldn't — accepted, it's a hard
  constraint, and the golden-set eval will catch it if candidate depth or model choice
  makes this unacceptable.
- Splitting eval work across two frameworks (DeepEval for CI-fast, Ragas for nightly-
  full) means two sets of framework-version pins and two APIs to keep current against
  their respective changelogs — accepted because the fast/full split is already
  mandated by `CLAUDE.md` (`evals-fast` vs `evals-full`), so the frameworks map onto an
  existing seam rather than creating a new one.
- Everything living in one Postgres instance (app data, pgvector chunks, LangGraph
  checkpoints) means retrieval-index maintenance (`just ingest`, HNSW rebuilds) and
  orchestration write load share the same free-tier resource ceiling. Mitigated by
  giving pgvector a read-only DB role for the agent's query path per the security rules,
  and by keeping ingest a separate offline CLI (`apps/ingest`) rather than an inline
  request-time operation.

## Consequences

- `packages/core` defines `Embedder`, `LexicalSearcher`, `Fuser`, and `Reranker`
  Protocols with zero I/O; `packages/retrieval` implements them against pgvector,
  Postgres FTS, RRF, and `sentence-transformers.CrossEncoder` respectively.
- `pyproject.toml` pins `sentence-transformers` (embedding + `CrossEncoder` reranking),
  `deepeval==4.1.5`, `ragas` at the version current at merge time (re-verify — Ragas's
  release cadence moves faster than this doc).
- New corpus ingestion must create/refresh the HNSW index explicitly as part of
  `just ingest`, not rely on an implicit default.
- Every new golden-set item needs a provenance entry (source, author, date, difficulty)
  per `CLAUDE.md`; citation-validity and abstention metrics apply to every item, not
  just a subset.
- Before implementation, re-confirm the exact pgvector HNSW build parameters
  (`m`, `ef_construction`) and the chosen `ef_search` against current pgvector docs —
  this ADR fixes the index *type*, not tuned parameters, which should be set from
  eval results on the actual corpus, not guessed upfront.
