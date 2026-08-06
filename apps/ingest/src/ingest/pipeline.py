from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector import Vector
from psycopg.types.json import Jsonb
from sentence_transformers import SentenceTransformer

from ingest.chunks import IngestChunk, build_chunks
from ingest.documents import DocumentSource
from ingest.fetch import fetch_document_xhtml
from ingest.parser import ParsedDocument, parse_document
from telemetry import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestStats:
    articles: int
    recitals: int
    chunks: int
    inserted: int


def _truncate(doc: ParsedDocument, max_items: int | None) -> ParsedDocument:
    if max_items is None:
        return doc
    return ParsedDocument(
        source=doc.source,
        articles=doc.articles[:max_items],
        recitals=doc.recitals[:max_items],
    )


def _upsert_chunks(
    conn: psycopg.Connection, chunks: list[IngestChunk], embeddings: list[list[float]]
) -> int:
    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cur.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, content, embedding, metadata)
                VALUES (%(chunk_id)s, %(document_id)s, %(content)s, %(embedding)s, %(metadata)s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "embedding": Vector(embedding),
                    "metadata": Jsonb(chunk.metadata),
                },
            )
    conn.commit()
    return len(chunks)


def ingest_document(
    source: DocumentSource,
    conn: psycopg.Connection,
    embedder_model: SentenceTransformer,
    cache_dir: Path,
    *,
    max_items: int | None = None,
    force_refresh: bool = False,
) -> IngestStats:
    """Fetch -> parse -> chunk -> embed -> idempotently upsert one document.
    `max_items` caps the number of articles/recitals parsed (used by the integration
    test to keep a real fetch fast, not a normal production knob)."""
    xhtml = fetch_document_xhtml(source, cache_dir, force_refresh=force_refresh)
    logger.info("ingest.document_fetched", source=source.slug, bytes=len(xhtml))

    doc = _truncate(parse_document(xhtml, source), max_items)
    logger.info(
        "ingest.articles_parsed",
        source=source.slug,
        articles=len(doc.articles),
        recitals=len(doc.recitals),
    )

    chunks = build_chunks(doc)
    logger.info("ingest.chunks_built", source=source.slug, chunks=len(chunks))

    if chunks:
        vectors = embedder_model.encode(
            [chunk.content for chunk in chunks], normalize_embeddings=True
        )
        embeddings = [vector.tolist() for vector in vectors]
    else:
        embeddings = []
    logger.info("ingest.chunks_embedded", source=source.slug, chunks=len(embeddings))

    inserted = _upsert_chunks(conn, chunks, embeddings) if chunks else 0
    logger.info("ingest.chunks_inserted", source=source.slug, chunks=inserted)

    return IngestStats(
        articles=len(doc.articles),
        recitals=len(doc.recitals),
        chunks=len(chunks),
        inserted=inserted,
    )
