import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(StrEnum):
    """Deliberately a separate taxonomy from api.graph.schemas.QueryIntent
    (factual_lookup / role_scoped_applicability / out_of_scope, no cross_reference)
    — packages/evals must not depend on apps/api (wrong dependency direction: an app
    may depend on a package for CI gating, never the reverse), and these classify
    golden-set questions for eval breakdown, not live queries for routing. Do not
    "helpfully" unify the two."""

    FACTUAL_LOOKUP = "factual_lookup"
    ROLE_SCOPED = "role_scoped"
    CROSS_REFERENCE = "cross_reference"
    OUT_OF_SCOPE = "out_of_scope"


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    author: str
    date: str
    source_reference: str


class GoldenItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    id: str
    question: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    question_type: QuestionType
    provenance: Provenance


def load_golden_set(path: Path) -> list[GoldenItem]:
    """One JSON object per line. A genuinely empty file returns []. A malformed line
    raises loudly — a bad golden-set item is a data-quality bug, not something to
    silently skip past."""
    text = path.read_text(encoding="utf-8")
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(GoldenItem.model_validate(json.loads(line)))
    return items
