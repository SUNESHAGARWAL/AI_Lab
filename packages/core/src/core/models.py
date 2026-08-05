from pydantic import BaseModel, ConfigDict, Field


class Filters(BaseModel):
    """Caller-expressed retrieval constraints. Adapters translate this into their own
    query language (e.g. a SQL WHERE clause) — this shape says only what a caller may
    ask for, not how it's satisfied."""

    model_config = ConfigDict(frozen=True)

    document_ids: list[str] | None = None
    tags: list[str] | None = None
    metadata_equals: dict[str, str] | None = None


class Query(BaseModel):
    """A single retrieval request."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    top_k: int = Field(default=10, gt=0)
    filters: Filters | None = None


class Chunk(BaseModel):
    """A unit of retrievable, citable corpus text. `id`/`document_id` are what
    citation-validity checks resolve back to source text."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A `Chunk` plus the relevance score that produced its rank in a result list."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
