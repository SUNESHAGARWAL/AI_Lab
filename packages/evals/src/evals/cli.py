import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from core.models import Query, ScoredChunk
from core.ports import Reranker, Retriever
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis
from retrieval.retriever import PgVectorRetriever

from evals.candidate_questions import SEED_QUESTIONS
from evals.candidates import generate_candidates, load_candidates, write_candidates
from evals.generation_run import build_generation_graph, run_golden_set_through_graph
from evals.generation_scorecard import (
    aggregate_generation_scorecard,
    render_generation_table,
    write_generation_report,
)
from evals.golden import GoldenItem, append_golden_items, load_golden_set
from evals.judge import DEFAULT_JUDGE_TIER, GatewayJudgeModel
from evals.judge_agreement import build_judge_agreement_report
from evals.promote import promote_to_golden_set
from evals.review import run_interactive_review
from evals.scorecard import RetrieveFn, aggregate, render_table, write_report
from llm import Gateway, Tier
from retrieval import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANKER_MODEL_NAME,
    CrossEncoderReranker,
    SentenceTransformerEmbedder,
    apply_migrations,
    create_pool,
)
from telemetry import get_logger

app = typer.Typer()
logger = get_logger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GOLDEN_SET = REPO_ROOT / "evals" / "datasets" / "retrieval_golden.jsonl"
DEFAULT_REPORTS_DIR = REPO_ROOT / "evals" / "reports"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "evals" / "datasets" / "candidates_for_review.jsonl"


async def _open_retriever() -> tuple[PgVectorRetriever, AsyncConnectionPool]:
    await apply_migrations(DATABASE_URL)
    pool = create_pool(DATABASE_URL)
    await pool.open(wait=True, timeout=10)
    embedder = SentenceTransformerEmbedder()
    return PgVectorRetriever(pool, embedder), pool


def build_retrieve_fn(
    retriever: Retriever, reranker: Reranker | None, rerank_pool_size: int
) -> RetrieveFn:
    """Builds the closure aggregate() calls per golden item. With no reranker, this
    is exactly PgVectorRetriever.retrieve() unchanged. With a reranker, it fetches a
    WIDER pool than top_k before reranking — reranking a list only as long as top_k
    could only reorder within it, never surface a chunk cosine similarity ranked
    just outside that range, which would defeat the point of comparing the two."""

    async def retrieve_fn(question: str, top_k: int) -> list[str]:
        if reranker is None:
            results = await retriever.retrieve(Query(text=question, top_k=top_k))
            return [r.chunk.id for r in results]

        pool_size = max(rerank_pool_size, top_k)
        query = Query(text=question, top_k=pool_size)
        candidates = await retriever.retrieve(query)
        reranked = await reranker.rerank(query, candidates)
        return [r.chunk.id for r in reranked[:top_k]]

    return retrieve_fn


async def _run_retrieval_async(
    golden_set: Path,
    k_values: list[int],
    output_dir: Path,
    rerank: bool,
    rerank_pool_size: int,
) -> None:
    golden_items = load_golden_set(golden_set)
    logger.info("evals.golden_set_loaded", path=str(golden_set), count=len(golden_items))

    retriever, pool = await _open_retriever()
    # Only load the cross-encoder model when actually reranking — no reason to pay
    # that cost for the default, pure-retrieval path.
    reranker = CrossEncoderReranker() if rerank else None
    try:
        retrieve_fn = build_retrieve_fn(retriever, reranker, rerank_pool_size)
        scorecard = await aggregate(
            golden_items,
            retrieve_fn,
            k_values,
            golden_set_path=str(golden_set),
            retriever_model=DEFAULT_MODEL_NAME,
            generated_at=datetime.now(UTC).isoformat(),
            reranked=rerank,
            reranker_model=DEFAULT_RERANKER_MODEL_NAME if rerank else None,
        )
    finally:
        await pool.close()

    json_path, md_path = write_report(scorecard, output_dir)
    logger.info(
        "evals.scorecard_written",
        json_path=str(json_path),
        md_path=str(md_path),
        scored_items=scorecard.scored_items,
        out_of_scope_excluded=scorecard.out_of_scope_excluded,
    )
    typer.echo(render_table(scorecard))


_GOLDEN_SET_OPTION = typer.Option(DEFAULT_GOLDEN_SET, help="Path to the golden-set JSONL file.")
_K_OPTION = typer.Option("1,3,5,10", help="Comma-separated list of k values.")
_OUTPUT_DIR_OPTION = typer.Option(DEFAULT_REPORTS_DIR, help="Directory to write the scorecard to.")
_RERANK_OPTION = typer.Option(
    False, "--rerank", help="Score reranked order instead of raw retrieval order."
)
_RERANK_POOL_SIZE_OPTION = typer.Option(
    20,
    "--rerank-pool-size",
    help="Candidates fetched per query before reranking (must exceed the largest k).",
)


@app.command("run-retrieval")
def run_retrieval(
    golden_set: Path = _GOLDEN_SET_OPTION,
    k: str = _K_OPTION,
    output_dir: Path = _OUTPUT_DIR_OPTION,
    rerank: bool = _RERANK_OPTION,
    rerank_pool_size: int = _RERANK_POOL_SIZE_OPTION,
) -> None:
    """Scores the real PgVectorRetriever against a golden set: recall@k, MRR@k,
    nDCG@k, no LLM/judge call — the deterministic layer-2 retrieval metrics. Pass
    --rerank to score the real CrossEncoderReranker's reordering instead of raw
    cosine order — the two runs are directly comparable via the scorecard's
    `reranked` field."""
    k_values = [int(part) for part in k.split(",")]
    asyncio.run(
        _run_retrieval_async(golden_set, k_values, output_dir, rerank, rerank_pool_size)
    )


async def _generate_candidates_async(output: Path, top_k: int, force: bool) -> None:
    retriever, pool = await _open_retriever()
    try:

        async def scored_retrieve_fn(question: str, k: int) -> list[ScoredChunk]:
            return await retriever.retrieve(Query(text=question, top_k=k))

        candidates = await generate_candidates(
            scored_retrieve_fn,
            SEED_QUESTIONS,
            top_k,
            generated_at=datetime.now(UTC).isoformat(),
        )
    finally:
        await pool.close()

    write_candidates(candidates, output, force=force)
    logger.info("evals.candidates_generated", path=str(output), count=len(candidates))


_CANDIDATES_OUTPUT_OPTION = typer.Option(
    DEFAULT_CANDIDATES_PATH, help="Path to write candidates_for_review.jsonl."
)
_TOP_K_OPTION = typer.Option(
    5, "--top-k", help="Number of candidate chunks to surface per question."
)
_FORCE_OPTION = typer.Option(
    False, "--force", help="Overwrite even if the existing file has reviewed items."
)


@app.command("generate-candidates")
def generate_candidates_command(
    output: Path = _CANDIDATES_OUTPUT_OPTION,
    top_k: int = _TOP_K_OPTION,
    force: bool = _FORCE_OPTION,
) -> None:
    """Runs ~40 grounded candidate questions through the REAL retriever and writes
    their real top-k results (chunk_id/score/content) to a review file. Never
    populates relevant_chunk_ids or sets verified=True — this harness proposes
    nothing; a human decides relevance by reading the real candidate_pool."""
    asyncio.run(_generate_candidates_async(output, top_k, force))


_REVIEW_PATH_OPTION = typer.Option(
    DEFAULT_CANDIDATES_PATH, help="Path to candidates_for_review.jsonl."
)


@app.command("review-candidates")
def review_candidates_command(path: Path = _REVIEW_PATH_OPTION) -> None:
    """Interactively review candidates_for_review.jsonl: for each item still
    needing a decision, shows the real candidate pool (full content) and records
    YOUR selection — never pre-fills or guesses relevant_chunk_ids. Saves after
    every single item, so nothing is lost if you stop partway."""
    items = load_candidates(path)
    summary = run_interactive_review(items, path)
    typer.echo(
        f"populated selections: {summary.populated}  "
        f"out-of-scope (intentionally empty): {summary.out_of_scope_empty}  "
        f"confirmed none (in-scope): {summary.confirmed_none}  "
        f"dropped: {summary.dropped}  "
        f"still undecided: {summary.still_undecided}"
    )


_PROMOTE_AUTHOR_OPTION = typer.Option(
    ..., "--author", help="Your name/handle, recorded in each promoted item's provenance."
)
_PROMOTE_CANDIDATES_OPTION = typer.Option(
    DEFAULT_CANDIDATES_PATH, help="Path to candidates_for_review.jsonl."
)
_PROMOTE_GOLDEN_SET_OPTION = typer.Option(
    DEFAULT_GOLDEN_SET, help="Path to the golden-set JSONL file to append to."
)


@app.command("promote-candidates")
def promote_candidates_command(
    author: str = _PROMOTE_AUTHOR_OPTION,
    candidates_path: Path = _PROMOTE_CANDIDATES_OPTION,
    golden_set: Path = _PROMOTE_GOLDEN_SET_OPTION,
) -> None:
    """Promotes already-reviewed candidates_for_review.jsonl items into the real
    golden set: populated selections and decided out_of_scope items only. Every
    relevant_chunk_ids value is copied verbatim from what you already selected in
    review-candidates — nothing new is decided here. Confirmed-none in-scope items
    are never silently promoted; they're reported for you to resolve by hand.
    Idempotent: ids already present in the golden set are skipped, not duplicated."""
    candidates = load_candidates(candidates_path)
    existing_golden = load_golden_set(golden_set)
    new_items, result = promote_to_golden_set(
        candidates,
        existing_golden,
        author=author,
        promotion_date=datetime.now(UTC).date().isoformat(),
    )
    if new_items:
        append_golden_items(new_items, golden_set)

    logger.info(
        "evals.candidates_promoted",
        promoted=len(result.promoted),
        already_in_golden_set=len(result.already_in_golden_set),
        needs_manual_decision=len(result.needs_manual_decision),
        dropped=result.dropped,
    )
    typer.echo(f"promoted: {len(result.promoted)}")
    typer.echo(f"already in golden set (skipped): {len(result.already_in_golden_set)}")
    typer.echo(f"dropped (excluded, not promoted): {result.dropped}")
    if result.needs_manual_decision:
        typer.echo(f"needs your manual decision ({len(result.needs_manual_decision)}):")
        for id_, question in result.needs_manual_decision:
            typer.echo(f"  {id_}: {question}")


async def _existing_chunk_ids(pool: AsyncConnectionPool) -> set[str]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT chunk_id FROM chunks")
        rows = await cur.fetchall()
    return {row[0] for row in rows}


def _select_sample(
    items: list[GoldenItem], sample: int | None, ids: str | None
) -> list[GoldenItem]:
    if ids:
        wanted = {i.strip() for i in ids.split(",") if i.strip()}
        selected = [item for item in items if item.id in wanted]
        missing = wanted - {item.id for item in selected}
        if missing:
            raise ValueError(f"--ids referenced unknown golden-set id(s): {sorted(missing)}")
        return selected
    if sample is None:
        return items
    return items[:sample]


async def _run_generation_eval_async(
    golden_set: Path, sample: int | None, judge_tier: Tier, output_dir: Path
) -> None:
    golden_items = load_golden_set(golden_set)
    selected = _select_sample(golden_items, sample, None)
    logger.info(
        "evals.generation_eval_started",
        golden_set=str(golden_set),
        total_items=len(golden_items),
        sample_size=len(selected),
    )

    retriever, pool = await _open_retriever()
    redis_client = Redis.from_url(REDIS_URL)
    gateway = Gateway(redis_client=redis_client)
    judge_model = GatewayJudgeModel(gateway, tier=judge_tier)
    try:
        graph = build_generation_graph(retriever, gateway)
        results = await run_golden_set_through_graph(selected, graph)
        if len(results) < len(selected):
            logger.warning(
                "evals.generation_items_skipped",
                requested=len(selected),
                completed=len(results),
                skipped=len(selected) - len(results),
            )
        existing_chunk_ids = await _existing_chunk_ids(pool)
        scorecard = await aggregate_generation_scorecard(
            results,
            gateway,
            judge_model,
            golden_set_path=str(golden_set),
            existing_chunk_ids=existing_chunk_ids,
            generated_at=datetime.now(UTC).isoformat(),
            total_golden_items=len(golden_items),
        )
    finally:
        await pool.close()

    json_path, md_path = write_generation_report(scorecard, output_dir)
    logger.info(
        "evals.generation_scorecard_written",
        json_path=str(json_path),
        md_path=str(md_path),
        sample_size=scorecard.sample_size,
    )
    typer.echo(render_generation_table(scorecard))


def _parse_judge_tier(value: str) -> Tier:
    try:
        return Tier(value)
    except ValueError as exc:
        valid = [t.value for t in Tier]
        raise typer.BadParameter(f"unknown tier {value!r} — must be one of {valid}") from exc


_GENERATION_GOLDEN_SET_OPTION = typer.Option(
    DEFAULT_GOLDEN_SET, help="Path to the golden-set JSONL file."
)
_GENERATION_SAMPLE_OPTION = typer.Option(
    None, "--sample", help="Score only the first N golden items. Default: all items."
)
_GENERATION_JUDGE_TIER_OPTION = typer.Option(
    DEFAULT_JUDGE_TIER.value, "--judge-tier", help="Gateway tier used for judge calls."
)
_GENERATION_OUTPUT_DIR_OPTION = typer.Option(
    DEFAULT_REPORTS_DIR, help="Directory to write the generation scorecard to."
)


@app.command("run-generation-eval")
def run_generation_eval(
    golden_set: Path = _GENERATION_GOLDEN_SET_OPTION,
    sample: int | None = _GENERATION_SAMPLE_OPTION,
    judge_tier: str = _GENERATION_JUDGE_TIER_OPTION,
    output_dir: Path = _GENERATION_OUTPUT_DIR_OPTION,
) -> None:
    """Layer 3: runs the real graph end-to-end per golden item and scores the
    generated answer — faithfulness/answer_relevancy (deepeval, judged through our
    gateway), citation_validity and appropriate_abstention (this package's own
    metrics), context_precision (deterministic, from golden-set ground truth). Makes
    real LLM calls — start with --sample 5 to check cost before running the full
    set, and read the judge-agreement-report output before trusting these numbers
    at scale."""
    asyncio.run(
        _run_generation_eval_async(golden_set, sample, _parse_judge_tier(judge_tier), output_dir)
    )


async def _judge_agreement_report_async(
    golden_set: Path, sample: int, ids: str | None, judge_tier: Tier, output_dir: Path
) -> None:
    golden_items = load_golden_set(golden_set)
    selected = _select_sample(golden_items, sample, ids)
    logger.info("evals.judge_agreement_started", sample_size=len(selected))

    retriever, pool = await _open_retriever()
    redis_client = Redis.from_url(REDIS_URL)
    gateway = Gateway(redis_client=redis_client)
    judge_model = GatewayJudgeModel(gateway, tier=judge_tier)
    try:
        graph = build_generation_graph(retriever, gateway)
        results = await run_golden_set_through_graph(selected, graph)
        report = await build_judge_agreement_report(results, gateway, judge_model)
    finally:
        await pool.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat().replace(":", "").replace("-", "").replace(".", "")
    report_path = output_dir / f"judge_agreement_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("evals.judge_agreement_written", path=str(report_path))
    typer.echo(f"judge agreement report written to {report_path}")
    typer.echo("This is a HUMAN VALIDATION step — read it before trusting run-generation-eval.")


_JUDGE_AGREEMENT_SAMPLE_OPTION = typer.Option(
    8, "--sample", help="Number of golden items (first N in file order) to include."
)
_JUDGE_AGREEMENT_IDS_OPTION = typer.Option(
    None, "--ids", help="Comma-separated golden-item ids to include, overriding --sample."
)


@app.command("judge-agreement-report")
def judge_agreement_report(
    golden_set: Path = _GENERATION_GOLDEN_SET_OPTION,
    sample: int = _JUDGE_AGREEMENT_SAMPLE_OPTION,
    ids: str | None = _JUDGE_AGREEMENT_IDS_OPTION,
    judge_tier: str = _GENERATION_JUDGE_TIER_OPTION,
    output_dir: Path = _GENERATION_OUTPUT_DIR_OPTION,
) -> None:
    """Runs a small hand-pickable set of golden items through the real graph and
    real judge, writing a human-readable Markdown report of every verdict alongside
    the answer/citations it judged. NOT an aggregate scorecard — read this by hand
    to confirm the judge is calling faithfulness/citation-support correctly before
    trusting run-generation-eval's numbers at scale."""
    asyncio.run(
        _judge_agreement_report_async(
            golden_set, sample, ids, _parse_judge_tier(judge_tier), output_dir
        )
    )


if __name__ == "__main__":
    app()
