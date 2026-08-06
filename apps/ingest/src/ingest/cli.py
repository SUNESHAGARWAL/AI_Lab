from pathlib import Path

import psycopg
import typer
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from ingest.config import Settings
from ingest.documents import DOCUMENTS
from ingest.pipeline import ingest_document
from retrieval import DEFAULT_MODEL_NAME, apply_migrations_sync
from telemetry import get_logger

app = typer.Typer()
logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


@app.command()
def status() -> None:
    """Report ingest CLI readiness. Corpus pipeline lands separately from this scaffold."""
    typer.echo("ingest CLI: skeleton only, no corpus pipeline wired up yet")


@app.command("ingest-corpus")
def ingest_corpus(
    force_refresh: bool = typer.Option(
        False, "--force-refresh", help="Refetch documents even if a cached copy exists."
    ),
    max_items: int | None = typer.Option(
        None,
        "--max-items",
        help="Cap the number of articles/recitals parsed per document (for manual testing).",
    ),
) -> None:
    """Fetches, parses, chunks, embeds, and idempotently upserts the EU AI Act and
    GDPR into the pgvector chunks table. Sync throughout, per CLAUDE.md's
    "sync only in apps/ingest" convention — this is a one-shot offline batch job,
    not part of the async request path."""
    settings = Settings()

    apply_migrations_sync(settings.database_url)

    embedder_model = SentenceTransformer(DEFAULT_MODEL_NAME, device="cpu")

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        for source in DOCUMENTS:
            stats = ingest_document(
                source,
                conn,
                embedder_model,
                DEFAULT_CACHE_DIR,
                max_items=max_items,
                force_refresh=force_refresh,
            )
            logger.info(
                "ingest.document_complete",
                source=source.slug,
                articles=stats.articles,
                recitals=stats.recitals,
                chunks=stats.chunks,
                inserted=stats.inserted,
            )


if __name__ == "__main__":
    app()
