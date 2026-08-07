# 0006 — context_precision = 0.167 is a chunk-grain artifact, not a retrieval bug

## Status

Accepted — 2026-08-07

## Context

The Layer 3 generation scorecard (`evals/reports/generation_scorecard_20260807T1425427288930000.md`,
36/36 golden items) shows every metric healthy except one:

```
appropriate_abstention: 36/36 correct
mean citation_validity: 0.9339
mean faithfulness:      0.9645
mean answer_relevancy:  0.9794
mean context_precision: 0.1667
```

`context_precision` (`packages/evals/src/evals/generation_metrics.py::context_precision`) is
computed deterministically per ADR 0004 point 3: `|context_chunk_ids ∩ relevant_chunk_ids| /
|context_chunk_ids|`, an exact-string-id match between the golden set's `relevant_chunk_ids`
and the chunk ids in `state["reranked_chunks"]` actually shown to the generator. A score this
low needed diagnosis before deciding whether it represents a real retrieval defect or an
artifact of how the metric counts.

The working hypothesis going in was that this project's "parent-document retrieval" chunking
(`apps/ingest/src/ingest/chunks.py::build_chunks` — every article gets a full-text parent
chunk, multi-paragraph articles additionally get paragraph-level leaf chunks carrying a
`parent_chunk_id` back to the article) was deliberately feeding full parent articles as
context, most of which don't match a golden set labeled at paragraph grain.

**Static trace ruled that hypothesis out.** `parent_chunk_id` is written into chunk metadata
at ingest time and is **never read again** — not by `packages/retrieval/src/retrieval/retriever.py`
(`PgVectorRetriever.retrieve` is a plain pgvector cosine-similarity `ORDER BY ... LIMIT top_k`
over one flat `chunks` table, no join or lookup on `parent_chunk_id`), not by
`apps/api/src/api/graph/nodes.py` (`retriever_node`/`reranker_node` pass chunks straight
through with no parent-fetch step). There is no parent-document-expansion step anywhere in the
runtime path, despite the ingest module's own "parent-document retrieval chunking" framing.

**Live diagnosis confirmed the actual mechanism instead.** A one-off script (reusing
`evals.generation_run.build_generation_graph` + `run_golden_set_through_graph`, no judge
calls) ran 3 golden items chosen for a single, unambiguous paragraph-level
`relevant_chunk_ids` label, and classified every one of the 5 chunks fed to the generator per
item as golden match / same-article sibling-or-parent / genuinely unrelated:

| item | golden question | context fed | golden match | same-article sibling/parent | genuinely unrelated |
|---|---|---|---|---|---|
| `cand-005` | Art 9 AI Act risk management system | 5 | 1 (`article:9:paragraph:2`) | 3 (`article:9`, `paragraph:1`, `paragraph:6`) | 1 (`article:8:paragraph:1`) |
| `cand-013` | Art 33 GDPR breach notification | 5 | 1 (`article:33:paragraph:1`) | 4 (`article:33`, `paragraph:2`, `paragraph:3`, `paragraph:5`) | 0 |
| `cand-016` | Art 26 AI Act deployer obligations | 5 | 1 (`article:26:paragraph:1`) | 4 (`article:26`, `paragraph:7`, `paragraph:9`, `paragraph:12`) | 0 |

**15 chunks total: 3 golden matches (20%, close to the aggregate 16.67%), 11 same-article
siblings/parent (73%), 1 genuinely unrelated chunk (7%,** `eu_ai_act:article:8:paragraph:1` —
adjacent-article content, on-topic to high-risk AI system compliance broadly, not Article 9
specifically).

The actual mechanism: `build_chunks` indexes both article-level and paragraph-level chunks as
**independent, competing vectors** in the same ANN search (no parent/child relationship
enforced at query time). A paragraph and its parent article overlap heavily in text, so both —
plus sibling paragraphs of the same article — score high cosine similarity for the same query
and cluster together in the reranked top-5. `context_precision`'s exact-id-match scoring has no
way to credit an on-topic sibling or parent as "relevant" when the golden label points at one
specific paragraph id.

**This is a labeling-grain / scoring-grain mismatch, not a retrieval defect**, and the rest of
Layer 3's numbers are the evidence: `citation_validity` (0.9339) and `faithfulness` (0.9645) are
judged against the *actual content* of every chunk shown to the generator — including the
same-article siblings `context_precision` scores as "irrelevant" — and both come back high. If
the reranker were genuinely surfacing off-topic context, faithfulness (does the answer only
claim what the shown context supports) and citation_validity (does each cited chunk actually
support the answer) would both be dragged down by that noise. They aren't. The generator is
working from correct signal; `context_precision`'s denominator just includes more than the
golden set can currently credit.

## Decision

1. **`context_precision` = 0.1667 is accepted as expected, not a defect**, given the diagnosis
   above. No retrieval, reranking, or ingestion change is being made as a result of this ADR.
2. **The metric itself is not being changed in this task.** A grain-aware version (e.g.
   crediting a same-article sibling/parent as partial or full credit, or normalizing both
   golden ids and retrieved ids to article grain before intersecting) is a real option but a
   separate, deliberate task — not bundled into this diagnosis.
3. **The golden set is not being changed in this task.** Re-labeling `relevant_chunk_ids` at
   article grain instead of paragraph grain would raise the score but would also lose the
   finer-grained ground truth `packages/evals/src/evals/candidates.py`'s human review process
   was built to produce — not done here without that trade-off being made deliberately.

## Known limitation / future work

`parent_chunk_id` (`apps/ingest/src/ingest/chunks.py`, chunk metadata) is currently **dead
infrastructure** — written at ingest time, read by nothing downstream. Two real follow-ups
this ADR deliberately leaves open rather than deciding:

- **Wire it in**: use `parent_chunk_id` to actually implement parent-document retrieval —
  e.g. retrieve on paragraph-level embeddings (finer-grained, better match precision) but
  expand each hit to its parent article before reranking/generation (more complete context for
  the generator). Would also let `context_precision` credit a query that hits a paragraph and
  is shown its (expected) parent, since the expansion would be deliberate rather than
  coincidental vector-space overlap. This is real generation-quality upside independent of the
  eval-metric question, not just a scoring fix.
- **Remove it**: if parent-document expansion is decided against (e.g. flat-chunk retrieval is
  already working, per the healthy faithfulness/citation_validity numbers above), delete the
  unused `parent_chunk_id` field and its docstring claim rather than carrying dead metadata.

No decision is made here on which of these two paths to take — flagged as a prioritization
call, not resolved by this diagnosis.

## Consequences

- Anyone reading the Layer 3 scorecard should read `context_precision` alongside
  `citation_validity`/`faithfulness`, not in isolation — a low `context_precision` with high
  `citation_validity`/`faithfulness` (this project's actual state) indicates a scoring-grain
  gap, not a quality problem, per this ADR's diagnosis. A low `context_precision` *alongside*
  low `citation_validity`/`faithfulness` would indicate the opposite and should be
  investigated as a real retrieval defect, not assumed to be this same artifact.
- If `parent_chunk_id` is ever wired into retrieval (first bullet above), this ADR's numbers
  become a stale baseline — that work should re-run `just evals-full` and compare against the
  0.1667 baseline recorded here.
- No code, dataset, or metric changed by this ADR — it is a diagnosis record only.
