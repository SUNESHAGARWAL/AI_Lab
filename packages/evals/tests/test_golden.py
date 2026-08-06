import json
from pathlib import Path

import pytest
from evals.golden import Difficulty, GoldenItem, QuestionType, append_golden_items, load_golden_set
from pydantic import ValidationError


def _item(**overrides) -> dict:
    base = {
        "id": "g-001",
        "question": "What does Article 6 say about high-risk classification?",
        "relevant_chunk_ids": ["eu_ai_act:article:6"],
        "difficulty": "easy",
        "question_type": "factual_lookup",
        "provenance": {
            "author": "jane",
            "date": "2026-08-06",
            "source_reference": "eu_ai_act:article:6",
        },
    }
    base.update(overrides)
    return base


def test_load_golden_set_round_trips_normal_item(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(_item()) + "\n", encoding="utf-8")

    items = load_golden_set(path)

    assert len(items) == 1
    item = items[0]
    assert item.id == "g-001"
    assert item.relevant_chunk_ids == ["eu_ai_act:article:6"]
    assert item.difficulty is Difficulty.EASY
    assert item.question_type is QuestionType.FACTUAL_LOOKUP
    assert item.provenance.author == "jane"
    assert item.schema_version == 1


def test_load_golden_set_handles_out_of_scope_item_with_empty_relevant_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    oos = _item(
        id="g-002",
        question="What is the capital of France?",
        relevant_chunk_ids=[],
        question_type="out_of_scope",
    )
    path.write_text(json.dumps(oos) + "\n", encoding="utf-8")

    items = load_golden_set(path)

    assert items[0].question_type is QuestionType.OUT_OF_SCOPE
    assert items[0].relevant_chunk_ids == []


def test_load_golden_set_empty_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_golden_set(path) == []


def test_load_golden_set_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(f"\n{json.dumps(_item())}\n\n", encoding="utf-8")

    assert len(load_golden_set(path)) == 1


def test_golden_item_rejects_missing_required_field() -> None:
    payload = _item()
    del payload["question"]

    with pytest.raises(ValidationError):
        GoldenItem.model_validate(payload)


def test_golden_item_rejects_invalid_question_type() -> None:
    payload = _item(question_type="not_a_real_type")

    with pytest.raises(ValidationError):
        GoldenItem.model_validate(payload)


def test_append_golden_items_creates_file_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    item = GoldenItem.model_validate(_item())

    append_golden_items([item], path)

    assert load_golden_set(path) == [item]


def test_append_golden_items_appends_without_clobbering_existing(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    first = GoldenItem.model_validate(_item(id="g-001"))
    second = GoldenItem.model_validate(_item(id="g-002"))

    append_golden_items([first], path)
    append_golden_items([second], path)

    loaded = load_golden_set(path)
    assert [i.id for i in loaded] == ["g-001", "g-002"]
