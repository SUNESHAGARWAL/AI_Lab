import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from core.models import ScoredChunk
from pydantic import BaseModel, ConfigDict, Field

from evals.candidate_questions import SeedQuestion
from evals.golden import Difficulty, QuestionType

ScoredRetrieveFn = Callable[[str, int], Awaitable[list[ScoredChunk]]]


class CandidatePoolItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    score: float
    content: str


class CandidateForReview(BaseModel):
    """Unverified — nothing in this shape has been read/confirmed by a human.
    relevant_chunk_ids/verified are never populated by generation code; they exist
    purely as fields for a human reviewer to fill in by hand."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    id: str
    question: str
    difficulty: Difficulty
    question_type: QuestionType
    candidate_pool: list[CandidatePoolItem]
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    verified: bool = False
    notes: str | None = None
    generated_at: str


async def generate_candidates(
    retrieve_fn: ScoredRetrieveFn,
    questions: list[SeedQuestion],
    top_k: int,
    generated_at: str,
) -> list[CandidateForReview]:
    candidates: list[CandidateForReview] = []
    for seed in questions:
        results = await retrieve_fn(seed.question, top_k)
        candidate_pool = [
            CandidatePoolItem(chunk_id=r.chunk.id, score=r.score, content=r.chunk.text)
            for r in results
        ]
        candidates.append(
            CandidateForReview(
                id=seed.id,
                question=seed.question,
                difficulty=seed.difficulty,
                question_type=seed.question_type,
                candidate_pool=candidate_pool,
                generated_at=generated_at,
            )
        )
    return candidates


def _has_review_progress(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get("verified") or item.get("relevant_chunk_ids") or item.get("notes"):
            return True
    return False


def write_candidates(
    candidates: list[CandidateForReview], path: Path, *, force: bool = False
) -> None:
    if path.exists() and not force and _has_review_progress(path):
        raise ValueError(
            f"{path} already has reviewed items (verified/relevant_chunk_ids/notes set) — "
            "refusing to overwrite in-progress review work. Pass --force to overwrite anyway."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(candidate.model_dump_json())
            f.write("\n")
