from dataclasses import dataclass, field

from ingest.parser import Article, ParsedDocument
from ingest.references import extract_cross_references


@dataclass(frozen=True)
class IngestChunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


def _base_metadata(source_slug: str, article: Article) -> dict[str, str]:
    metadata = {"source": source_slug, "article": article.number}
    if article.chapter_number is not None:
        metadata["chapter"] = article.chapter_number
    if article.section_number is not None:
        metadata["section"] = article.section_number
    return metadata


def _article_chunk(source_slug: str, article: Article) -> IngestChunk:
    header = f"Article {article.number} — {article.title}"
    content = f"{header}\n\n{article.full_text}"
    metadata = _base_metadata(source_slug, article)
    refs = extract_cross_references(article.full_text)
    if refs:
        metadata["cross_references"] = ",".join(refs)
    return IngestChunk(
        chunk_id=f"{source_slug}:article:{article.number}",
        document_id=source_slug,
        content=content,
        metadata=metadata,
    )


def _paragraph_chunks(
    source_slug: str, article: Article, parent_chunk_id: str
) -> list[IngestChunk]:
    chunks: list[IngestChunk] = []
    for index, paragraph in enumerate(article.paragraphs, start=1):
        para_number = paragraph.number or str(index)
        header = f"Article {article.number}({para_number}) — {article.title}"
        content = f"{header}\n\n{paragraph.text}"
        metadata = _base_metadata(source_slug, article)
        metadata["paragraph"] = para_number
        metadata["parent_chunk_id"] = parent_chunk_id
        refs = extract_cross_references(paragraph.text)
        if refs:
            metadata["cross_references"] = ",".join(refs)
        chunks.append(
            IngestChunk(
                chunk_id=f"{source_slug}:article:{article.number}:paragraph:{para_number}",
                document_id=source_slug,
                content=content,
                metadata=metadata,
            )
        )
    return chunks


def build_chunks(doc: ParsedDocument) -> list[IngestChunk]:
    """Parent-document retrieval chunking: every Article always gets one article-level
    "parent" chunk (full text, precise enough on its own for single-paragraph
    articles). Multi-paragraph articles additionally get one paragraph-level "leaf"
    chunk each, small/precise for embedding, carrying parent_chunk_id back to the
    article-level chunk. Recitals are single-level — no parent/child split."""
    chunks: list[IngestChunk] = []
    slug = doc.source.slug

    for article in doc.articles:
        article_chunk = _article_chunk(slug, article)
        chunks.append(article_chunk)
        if len(article.paragraphs) > 1:
            chunks.extend(_paragraph_chunks(slug, article, article_chunk.chunk_id))

    for recital in doc.recitals:
        header = f"Recital {recital.number}"
        content = f"{header}\n\n{recital.text}"
        metadata = {"source": slug, "recital": recital.number}
        refs = extract_cross_references(recital.text)
        if refs:
            metadata["cross_references"] = ",".join(refs)
        chunks.append(
            IngestChunk(
                chunk_id=f"{slug}:recital:{recital.number}",
                document_id=slug,
                content=content,
                metadata=metadata,
            )
        )

    return chunks
