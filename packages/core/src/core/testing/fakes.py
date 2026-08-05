import hashlib
from collections.abc import Callable

from core.models import Chunk, Filters, Query, ScoredChunk

_EMBEDDING_DIMENSIONS = 16


def _hash_to_vector(text: str, dimensions: int = _EMBEDDING_DIMENSIONS) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [
        (int.from_bytes(digest[i : i + 2], "big") / 65535.0) * 2 - 1
        for i in range(0, dimensions * 2, 2)
    ]


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for a real embedding model. Same text
    always yields the same vector; no semantic similarity is implied — only useful for
    testing embedding-consumer plumbing, not retrieval quality."""

    async def embed_query(self, text: str) -> list[float]:
        return _hash_to_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vector(text) for text in texts]


def _passes_filters(chunk: Chunk, filters: Filters | None) -> bool:
    if filters is None:
        return True
    if filters.document_ids is not None and chunk.document_id not in filters.document_ids:
        return False
    if filters.tags is not None:
        chunk_tags = set(chunk.metadata.get("tags", "").split(","))
        if not chunk_tags.intersection(filters.tags):
            return False
    if filters.metadata_equals is not None:
        for key, value in filters.metadata_equals.items():
            if chunk.metadata.get(key) != value:
                return False
    return True


def _term_match_score(query_text: str, chunk_text: str) -> float:
    terms = [term for term in query_text.lower().split() if term]
    haystack = chunk_text.lower()
    return float(sum(haystack.count(term) for term in terms))


class FakeRetriever:
    """In-memory `Retriever` over a fixed corpus, scored by naive case-insensitive term
    counting. No randomness — same query against the same corpus always returns the
    same ranking, so tests can assert on exact output."""

    def __init__(self, corpus: list[Chunk]) -> None:
        self._corpus = list(corpus)

    async def retrieve(self, query: Query) -> list[ScoredChunk]:
        candidates = (c for c in self._corpus if _passes_filters(c, query.filters))
        scored = [
            ScoredChunk(chunk=chunk, score=_term_match_score(query.text, chunk.text))
            for chunk in candidates
        ]
        matched = [sc for sc in scored if sc.score > 0]
        matched.sort(key=lambda sc: sc.score, reverse=True)
        return matched[: query.top_k]


class FakeReranker:
    """In-memory `Reranker`. Defaults to identity passthrough; pass `score_fn` to
    control rerank order explicitly in a test."""

    def __init__(self, score_fn: Callable[[Query, Chunk], float] | None = None) -> None:
        self._score_fn = score_fn

    async def rerank(self, query: Query, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if self._score_fn is None:
            return list(candidates)
        rescored = [
            ScoredChunk(chunk=sc.chunk, score=self._score_fn(query, sc.chunk)) for sc in candidates
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored
