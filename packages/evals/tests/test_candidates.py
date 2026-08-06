import json
from pathlib import Path

import pytest
from core.models import Chunk, ScoredChunk
from evals.candidate_questions import SeedQuestion
from evals.candidates import generate_candidates, write_candidates
from evals.golden import Difficulty, QuestionType


def _seed(id_: str, question: str) -> SeedQuestion:
    return SeedQuestion(
        id=id_,
        question=question,
        difficulty=Difficulty.EASY,
        question_type=QuestionType.FACTUAL_LOOKUP,
        grounding_note="test",
    )


def _fake_scored_retrieve_fn(pool: list[ScoredChunk]):
    async def retrieve_fn(question: str, k: int) -> list[ScoredChunk]:
        return pool[:k]

    return retrieve_fn


@pytest.mark.asyncio
async def test_generate_candidates_never_populates_relevant_chunk_ids_or_verified() -> None:
    pool = [
        ScoredChunk(chunk=Chunk(id="c1", document_id="doc", text="content one"), score=0.9),
        ScoredChunk(chunk=Chunk(id="c2", document_id="doc", text="content two"), score=0.5),
    ]
    seeds = [_seed("cand-001", "what is x?")]

    candidates = await generate_candidates(
        _fake_scored_retrieve_fn(pool), seeds, top_k=2, generated_at="2026-08-06T00:00:00Z"
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.relevant_chunk_ids == []
    assert candidate.verified is False
    assert len(candidate.candidate_pool) == 2
    assert candidate.candidate_pool[0].chunk_id == "c1"
    assert candidate.candidate_pool[0].content == "content one"


@pytest.mark.asyncio
async def test_generate_candidates_respects_top_k() -> None:
    pool = [
        ScoredChunk(chunk=Chunk(id=f"c{i}", document_id="doc", text="x"), score=1.0 - i * 0.1)
        for i in range(5)
    ]
    seeds = [_seed("cand-001", "q")]

    candidates = await generate_candidates(
        _fake_scored_retrieve_fn(pool), seeds, top_k=3, generated_at="2026-08-06T00:00:00Z"
    )

    assert len(candidates[0].candidate_pool) == 3


def test_write_candidates_refuses_overwrite_when_review_progress_exists(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    reviewed_item = {
        "schema_version": 1,
        "id": "cand-001",
        "question": "q",
        "difficulty": "easy",
        "question_type": "factual_lookup",
        "candidate_pool": [],
        "relevant_chunk_ids": ["eu_ai_act:article:1"],
        "verified": True,
        "notes": None,
        "generated_at": "2026-08-06T00:00:00Z",
    }
    path.write_text(json.dumps(reviewed_item) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_candidates([], path, force=False)


def test_write_candidates_force_overwrites_reviewed_file(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    reviewed_item = {
        "schema_version": 1,
        "id": "cand-001",
        "question": "q",
        "difficulty": "easy",
        "question_type": "factual_lookup",
        "candidate_pool": [],
        "relevant_chunk_ids": ["eu_ai_act:article:1"],
        "verified": True,
        "notes": None,
        "generated_at": "2026-08-06T00:00:00Z",
    }
    path.write_text(json.dumps(reviewed_item) + "\n", encoding="utf-8")

    write_candidates([], path, force=True)

    assert path.read_text(encoding="utf-8") == ""


def test_write_candidates_allows_overwrite_when_nothing_reviewed_yet(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    unreviewed_item = {
        "schema_version": 1,
        "id": "cand-001",
        "question": "q",
        "difficulty": "easy",
        "question_type": "factual_lookup",
        "candidate_pool": [],
        "relevant_chunk_ids": [],
        "verified": False,
        "notes": None,
        "generated_at": "2026-08-06T00:00:00Z",
    }
    path.write_text(json.dumps(unreviewed_item) + "\n", encoding="utf-8")

    write_candidates([], path, force=False)  # should not raise

    assert path.read_text(encoding="utf-8") == ""
