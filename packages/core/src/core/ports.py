from typing import Protocol, runtime_checkable

from core.models import Query, ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Adapters may use different code paths for queries vs.
    documents (asymmetric models like BGE/e5 apply different instructions/prefixes to
    each) — both methods must still place their output in the same vector space, so a
    query embedding and a document embedding are directly comparable via cosine/L2."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns one vector."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document/chunk texts. Returns one vector per input text,
        in the same order as `texts` — `len(result) == len(texts)`."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Finds candidate chunks for a query. Implementation-agnostic: an adapter may be
    vector-only, lexical-only, or a hybrid fusion of both — the port only guarantees
    the shape of the result, not the retrieval strategy behind it."""

    async def retrieve(self, query: Query) -> list[ScoredChunk]:
        """Returns at most `query.top_k` chunks, ordered by descending `score`, honoring
        `query.filters` if set. An empty list is a valid "no matches" result, never an
        error — callers (and abstention logic) rely on this distinction."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Rescoring a fixed candidate set. Kept deliberately narrow: a reranker rescores
    and reorders, it never adds or removes candidates — truncating to a final top-k is
    the caller's responsibility (slice the returned list)."""

    async def rerank(self, query: Query, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """Returns exactly the same set of chunks as `candidates`, rescored against
        `query` and reordered by descending new `score`. `len(result) == len(candidates)`
        and every chunk in `candidates` appears exactly once in the result."""
        ...
