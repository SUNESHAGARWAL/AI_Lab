/** A cited chunk, narrowed from the reranker's node_completed payload — see
 * api.graph.events.reranker_payload, which dumps core.models.ScoredChunk as
 * {chunk: {id, document_id, text, metadata}, score}. Every id the generator can cite is
 * validated server-side against exactly this set (api.graph.nodes._validate_citations),
 * so a citation id always resolves here on the happy path; callers still handle a miss
 * defensively rather than assuming it. */
export interface CitedChunk {
  id: string;
  documentId: string;
  text: string;
  metadata: Record<string, string>;
}

function asCitedChunk(raw: unknown): CitedChunk | null {
  if (typeof raw !== "object" || raw === null || !("chunk" in raw)) return null;
  const chunk = (raw as { chunk: unknown }).chunk;
  if (typeof chunk !== "object" || chunk === null) return null;
  const { id, document_id, text, metadata } = chunk as Record<string, unknown>;
  if (typeof id !== "string" || typeof document_id !== "string" || typeof text !== "string") return null;
  return {
    id,
    documentId: document_id,
    text,
    metadata: typeof metadata === "object" && metadata !== null ? (metadata as Record<string, string>) : {},
  };
}

/** Builds an id -> chunk lookup from the reranker payload's `reranked` array — the exact
 * chunk set the generator saw and cited from. */
export function buildChunkIndex(reranked: unknown[] | undefined): Map<string, CitedChunk> {
  const index = new Map<string, CitedChunk>();
  for (const raw of reranked ?? []) {
    const chunk = asCitedChunk(raw);
    if (chunk) index.set(chunk.id, chunk);
  }
  return index;
}

/** "Art. 6(2)" / "GDPR Art. 4" / "Recital 12" / "GDPR Recital 47" — built from
 * apps/ingest/src/ingest/chunks.py's metadata shape: {source: "eu_ai_act"|"gdpr",
 * article, paragraph?} for articles, {source, recital} for recitals. GDPR gets a
 * document prefix since the AI Act is this system's implicit default corpus; the AI
 * Act doesn't need one. */
export function citationLabel(metadata: Record<string, string>): string {
  const prefix = metadata.source === "gdpr" ? "GDPR " : "";
  if (metadata.recital) return `${prefix}Recital ${metadata.recital}`;
  if (metadata.article) {
    const paragraph = metadata.paragraph ? `(${metadata.paragraph})` : "";
    return `${prefix}Art. ${metadata.article}${paragraph}`;
  }
  return `${prefix}source`;
}

/** Chunk text already carries a human header line — see _article_chunk/_paragraph_chunks/
 * build_chunks in apps/ingest/src/ingest/chunks.py, e.g. "Article 6(2) — <title>\n\n<body>"
 * or "Recital 12\n\n<body>". Split on the first blank line rather than re-deriving the
 * header from metadata, so the panel shows exactly what's in the corpus. */
export function splitChunkText(text: string, fallbackHeader: string): { header: string; body: string } {
  const boundary = text.indexOf("\n\n");
  if (boundary === -1) return { header: fallbackHeader, body: text };
  return { header: text.slice(0, boundary), body: text.slice(boundary + 2) };
}
