# 0003 — Cross-encoder reranking: built, evaluated, deferred from the default path

## Status

Accepted — 2026-08-06

## Context

ADR 0002 named local `CrossEncoder` reranking a "hard requirement" from `CLAUDE.md`
and specified `cross-encoder/ms-marco-MiniLM-L-6-v2` as the default model. When the
adapter was actually built (`packages/retrieval/src/retrieval/reranker.py`), the
model choice was revisited against current model cards and changed to
`BAAI/bge-reranker-base` — smaller than `bge-reranker-v2-m3` (0.3B vs 0.6B params),
already bilingual EN/ZH so no English-accuracy loss for this corpus, and consistent
with the same free-tier/CPU-only reasoning ADR 0002 already applied to the embedding
model. That divergence from ADR 0002's named model is noted here for the record; it
doesn't change ADR 0002's underlying requirement that reranking be local, never a
hosted API.

`CLAUDE.md`'s own rule — "changing retrieval without re-running `just evals-fast` is
not allowed" — meant the adapter couldn't just ship enabled on the strength of its
own unit tests (which only prove it *functions*: reorders, doesn't drop/add
candidates). Once the retrieval golden set was populated with 36 human-reviewed
items (30 scored, 6 `out_of_scope`) and `run-retrieval` gained a `--rerank` flag
(retrieves a pool of 20 candidates per query, reranks, then scores the reranked
top-k — wider than the final k specifically so the reranker has room to actually
move a chunk into the scored range, not just reorder within it), a real before/after
comparison became possible for the first time.

**Measured result**, `BAAI/bge-small-en-v1.5` retrieval vs. the same retrieval
followed by `BAAI/bge-reranker-base` reranking, both against the same 30 scored
golden-set items:

| k | recall@k (before → after) | MRR@k (before → after) | nDCG@k (before → after) |
|---|---|---|---|
| 1 | 0.650 → 0.550 | 0.667 → 0.567 | 0.667 → 0.567 |
| 3 | 0.967 → 0.933 | 0.789 → 0.728 | 0.834 → 0.781 |
| 5 | 1.000 → 0.933 | 0.796 → 0.728 | 0.847 → 0.781 |
| 10 | 1.000 → 0.933 | 0.796 → 0.728 | 0.847 → 0.781 |

Reranking degraded every metric at every k, including recall@10 — the reranker
demoted a correctly-retrieved chunk out of the scored top-10 for at least one query,
which is possible precisely because it reranks a wider pool (20) before truncating.

**Working hypothesis for why a general-domain reranker underperforms here** (not
proven, offered as the reasoning behind the decision below, not as established
fact):
- `bge-reranker-base` is trained on general web/QA-style relevance judgments, not
  fine-tuned on legal/regulatory text — this corpus's dense cross-referencing,
  formally defined terms, and Article/paragraph/point structure differ substantially
  from the passage-relevance signal the model learned from.
- Article-level chunks in this corpus can be long (e.g. AI Act Article 6 runs 8
  paragraphs); `CrossEncoder`'s default 512-token max length (matched explicitly in
  `CrossEncoderReranker`'s `DEFAULT_MAX_LENGTH`) truncates longer chunks before
  scoring, so the cross-encoder may judge relevance from a truncated fragment that
  doesn't contain the actually-relevant paragraph.
- Many golden-set questions are close paraphrases of a chunk's own header/opening
  sentence (e.g. "What is the subject matter of the AI Act according to Article 1?"
  vs. "Article 1 — Subject matter"), which already favors `bge-small-en-v1.5`'s
  cosine similarity (tuned with its query-instruction prefix) — a general-purpose
  reranker has no special edge on this literal-citation query style and can
  introduce noise instead.
- 30 scored items is a small sample — a handful of flipped rankings swings recall@1
  by several points. This is the best real signal available today, not a
  statistically airtight verdict.

## Decision

1. **`apps/api/src/api/main.py` uses an identity-passthrough `FakeReranker()` in the
   default graph path, not `CrossEncoderReranker()`.** The graph's `reranker` node
   (`make_reranker_node` in `apps/api/src/api/graph/nodes.py`) is unchanged and
   still calls `reranker.rerank(query, candidates)` — with an identity reranker that
   call returns `candidates` unmodified, so the live product serves raw
   `PgVectorRetriever` cosine order, matching the retrieval-only side of the
   measured comparison above.
2. **`CrossEncoderReranker` and the `--rerank` eval flag stay in the codebase**,
   fully tested (`packages/retrieval/tests/test_reranker_integration.py`,
   `packages/evals/tests/test_cli_rerank.py`). This was the correct investment
   regardless of today's numbers — the eval-first workflow (build → measure →
   decide, not build → assume → ship) is what caught the regression before it
   reached the live path, and the same flag is what any future model swap gets
   measured through.
3. **Re-enabling reranking in `main.py` requires re-running `run-retrieval --rerank`
   against the golden set and confirming an improvement**, per `CLAUDE.md`'s
   eval-gates-retrieval-changes rule — not a general belief that reranking usually
   helps.

## Alternatives considered

- **Enable reranking anyway, on the general expectation that cross-encoders improve
  ranking quality.** Rejected: directly contradicts `CLAUDE.md`'s "the golden set is
  the source of truth for quality" and ADR 0002's own "changing retrieval without
  re-running `just evals-fast` is not allowed" — a change that measurably degrades
  the golden set cannot ship enabled just because it's usually a good idea.
- **Immediately try a different off-the-shelf reranker (e.g. `bge-reranker-v2-m3`).**
  Deferred, not rejected: the earlier model-choice reasoning (smaller model,
  English-only corpus, no accuracy loss from language coverage) still holds, and a
  bigger general-purpose model doesn't obviously address the more likely root cause
  (domain mismatch, chunk-length truncation) — not clearly a better next experiment
  than a domain-tuned model.
- **Increase `max_length`/`--rerank-pool-size` and re-test before deciding.** A
  legitimate, cheap follow-up experiment — noted under Consequences, not run here;
  the task at hand was to record the decision, not keep iterating on it.
- **Delete the reranker adapter entirely.** Rejected: the adapter, its tests, and
  the eval flag are correct engineering investment independent of today's result —
  only the default-path enablement decision changes.

## Trade-offs

- The live graph currently has one fewer ranking-quality lever active than ADR 0002
  originally envisioned. Accepted: shipping a measured regression would be worse
  than shipping without it.
- `apps/api`'s startup no longer loads the ~0.3B-parameter reranker model, which
  incidentally reduces cold-start time and memory — a real but secondary benefit of
  this decision, not the reason for it.
- `ADR 0002`'s "hard requirement" framing for reranking is superseded in practice:
  local (never hosted) reranking capability is still required and built; *always-on
  in the default path* is not, and is now gated on evidence the same way any other
  retrieval change is.

## Consequences

- Future reranker work should start from `uv run --package evals python -m evals.cli run-retrieval --rerank`
  against the current golden set, not from re-deriving this ADR's numbers by hand.
- Promising next experiments, in rough order of cost: (a) raise
  `CrossEncoderReranker`'s `max_length`/`--rerank-pool-size` and re-measure, in case
  truncation is the dominant cause; (b) evaluate a reranker with legal/regulatory
  domain exposure if one becomes available on the free tier; (c) grow the golden set
  beyond 30 scored items before treating any future rerank result as conclusive.
- Any future change to `apps/api/src/api/main.py`'s reranker wiring must cite a
  `run-retrieval --rerank` result in its commit message, per `CLAUDE.md`'s
  provenance expectations for retrieval-affecting changes.

---

## Addendum — 2026-08-11: re-tested against a real-world failure, decision unchanged

**What prompted the revisit.** The deployed demo abstained on a plainly answerable
question: *"Who must appoint a data protection officer under GDPR?"* The abstention
itself was correct behaviour — the generator was never shown the governing text — but
the retrieval behind it was not.

**Diagnosis.** `gdpr:article:37:paragraph:1` *is* indexed. Dense cosine ranks it **9th**
for that phrasing, and the whole top-10 sits inside a **0.026** score band: Articles 37
and 38 are eight paragraphs all about DPOs, and `bge-small-en-v1.5` cannot separate them.
The `DEFAULT_RERANK_TOP_N = 5` cut then drops the one paragraph that answers the
question. Re-scoring those same candidates with `CrossEncoderReranker` in isolation moves
the target from **rank 9 to rank 2** — i.e. exactly the failure this ADR's reranker is
supposed to fix.

**The trigger is narrower than it first looked.** The golden set already contained
`cand-022`, targeting the same chunk in the statute's own vocabulary ("*designate* … under
Article 37"), which retrieves at rank 2. Removing the article number keeps it at rank 2;
swapping the verb *designate* → *appoint* is what drops it to 9. But this is not a general
vocabulary-gap problem: everyday paraphrases of two other golden items ("privacy impact
assessment" for `article:35:paragraph:3`, "data leak" for `article:33:paragraph:1`)
retrieve at ranks 2 and 1. The failure needs **both** an everyday synonym *and* a dense
cluster of near-identical sibling paragraphs competing for the same slots.

**What was added.** `cand-041` — the failing phrasing, `factual_lookup`, `hard`, targeting
`gdpr:article:37:paragraph:1`. It is deliberately kept distinct from `cand-022` so the
breakdown shows the phrasing sensitivity directly. The paraphrases that already rank 1–2
were deliberately *not* added: a golden item that passes trivially guards nothing and
still costs eval tokens.

**Re-run, 37 items / 31 scored, `--rerank-pool-size 20`:**

```
   k |   recall@k dense →  rerank |     nDCG@k dense →  rerank
------------------------------------------------------------
   1 |     0.6290 →  0.5323       |     0.6452 →  0.5484
   3 |     0.9355 →  0.9355       |     0.8072 →  0.7757
   5 |     0.9677 →  0.9355       |     0.8197 →  0.7757
  10 |     1.0000 →  0.9355       |     0.8294 →  0.7757
```

**Verdict: the reranker stays off.** It loses on every metric except a tie at recall@3 —
*with the case that motivated the revisit now in the set*. Note recall@10 falling
1.0000 → 0.9355: because `--rerank` fetches a wider pool (20) before truncating, this is
the cross-encoder actively demoting two targets below rank 10, not a harness artifact. It
fixes `cand-041` and breaks more than it fixes.

**`cand-041` is therefore left failing on purpose.** It marks the boundary of what this
retrieval stack handles, in the one place that cannot be quietly forgotten. Per this
ADR's own Consequences list, the open options remain (a) `max_length`/pool-size tuning,
(b) a domain-exposed reranker, (c) a larger golden set — plus (d) hybrid lexical
retrieval, which ADR 0002's title anticipates but which is not currently built.

### Follow-up, same day: what `cand-041` actually exposed

The item was renumbered `cand-026` → `cand-041`: `cand-026` was already taken in
`evals/datasets/candidates_for_review.jsonl` by a different question, and
`promote_to_golden_set` skips any candidate whose id is already in the golden set — so
the collision would have silently made the real `cand-026` unpromotable forever.

Two Layer 3 runs over the 37-item set, identical retrieval, differing only in the
generator/critic system prompts (the untrusted-input hardening added for prompt
injection):

```
                      abstained  faithfulness  citation_validity  context_precision
before hardening         no          1.00            1.00               0.00
after  hardening         yes          —              0.00               0.00
```

`context_precision = 0.0` in both runs: `gdpr:article:37:paragraph:1` was never among the
five chunks the generator saw. Before hardening the model answered anyway, from the
neighbouring DPO paragraphs, and the judge scored it **1.0 on faithfulness, answer
relevancy and citation validity** — internally faithful to text that does not answer the
question. `appropriate_abstention` counted that as correct, so the aggregate read 37/37.

Three consequences worth keeping:

1. **A perfect abstention score can hide a wrong answer.** The metric only asks whether
   the abstain flag matched expectation; it never asks whether an answer was right. Here
   the flattering run was the incorrect one.
2. **`context_precision` was the only honest signal, and ADR 0006 is narrower than it
   reads.** A low *mean* is a dual-grain artifact, as documented. A **0.0 on one item** is
   not grain — it means nothing relevant was retrieved, and averaging destroys that
   distinction.
3. **The planner does not rescue this query.** An earlier hypothesis was that its rewrite
   normalises "appoint" to the statute's "designate". Measured directly: 0/5 runs
   rewrote it at all — the question passes through verbatim, so retrieval misses every
   time. The hypothesis was wrong.

After hardening, the honest outcome (abstain) is scored as a false abstention and the
headline becomes 36/37. That is the correct trade for this product, and `cand-041` stays
in the set failing on purpose.
