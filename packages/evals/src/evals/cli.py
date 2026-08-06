import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from core.models import Query, ScoredChunk
from psycopg_pool import AsyncConnectionPool
from retrieval.retriever import PgVectorRetriever

from evals.candidate_questions import SEED_QUESTIONS
from evals.candidates import generate_candidates, write_candidates
from evals.golden import load_golden_set
from evals.scorecard import aggregate, render_table, write_report
from retrieval import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbedder,
    apply_migrations,
    create_pool,
)
from telemetry import get_logger

app = typer.Typer()
logger = get_logger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app")
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


async def _run_retrieval_async(golden_set: Path, k_values: list[int], output_dir: Path) -> None:
    golden_items = load_golden_set(golden_set)
    logger.info("evals.golden_set_loaded", path=str(golden_set), count=len(golden_items))

    retriever, pool = await _open_retriever()
    try:

        async def retrieve_fn(question: str, top_k: int) -> list[str]:
            results = await retriever.retrieve(Query(text=question, top_k=top_k))
            return [r.chunk.id for r in results]

        scorecard = await aggregate(
            golden_items,
            retrieve_fn,
            k_values,
            golden_set_path=str(golden_set),
            retriever_model=DEFAULT_MODEL_NAME,
            generated_at=datetime.now(UTC).isoformat(),
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


@app.command("run-retrieval")
def run_retrieval(
    golden_set: Path = _GOLDEN_SET_OPTION,
    k: str = _K_OPTION,
    output_dir: Path = _OUTPUT_DIR_OPTION,
) -> None:
    """Scores the real PgVectorRetriever against a golden set: recall@k, MRR@k,
    nDCG@k, no LLM/judge call — the deterministic layer-2 retrieval metrics."""
    k_values = [int(part) for part in k.split(",")]
    asyncio.run(_run_retrieval_async(golden_set, k_values, output_dir))


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


if __name__ == "__main__":
    app()
