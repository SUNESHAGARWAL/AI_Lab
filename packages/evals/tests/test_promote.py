from evals.candidates import CandidateForReview, CandidatePoolItem
from evals.golden import Difficulty, GoldenItem, Provenance, QuestionType
from evals.promote import promotable_candidates, promote_to_golden_set
from evals.review import DROPPED_NOTES, NONE_NOTES


def _candidate(
    id_: str,
    question_type: QuestionType,
    *,
    relevant_chunk_ids: list[str] | None = None,
    verified: bool = True,
    notes: str | None = None,
) -> CandidateForReview:
    return CandidateForReview(
        id=id_,
        question=f"question for {id_}",
        difficulty=Difficulty.EASY,
        question_type=question_type,
        candidate_pool=[CandidatePoolItem(chunk_id=f"{id_}-c1", score=0.9, content="content")],
        verified=verified,
        relevant_chunk_ids=relevant_chunk_ids or [],
        notes=notes,
        generated_at="2026-08-06T00:00:00Z",
    )


def _golden(id_: str) -> GoldenItem:
    return GoldenItem(
        id=id_,
        question=f"question for {id_}",
        relevant_chunk_ids=["x"],
        difficulty=Difficulty.EASY,
        question_type=QuestionType.FACTUAL_LOOKUP,
        provenance=Provenance(author="someone", date="2026-08-01", source_reference="x"),
    )


# ---------------------------------------------------------------------------
# promotable_candidates
# ---------------------------------------------------------------------------


def test_promotable_candidates_splits_correctly() -> None:
    populated = _candidate("c1", QuestionType.FACTUAL_LOOKUP, relevant_chunk_ids=["x"])
    out_of_scope = _candidate("c2", QuestionType.OUT_OF_SCOPE)
    confirmed_none = _candidate("c3", QuestionType.ROLE_SCOPED, notes=NONE_NOTES)
    dropped = _candidate("c4", QuestionType.FACTUAL_LOOKUP, notes=DROPPED_NOTES)

    promotable, needs_manual = promotable_candidates(
        [populated, out_of_scope, confirmed_none, dropped]
    )

    assert {c.id for c in promotable} == {"c1", "c2"}
    assert {c.id for c in needs_manual} == {"c3"}


def test_promotable_candidates_never_includes_confirmed_none() -> None:
    confirmed_none = _candidate("c1", QuestionType.CROSS_REFERENCE, notes=NONE_NOTES)

    promotable, needs_manual = promotable_candidates([confirmed_none])

    assert promotable == []
    assert [c.id for c in needs_manual] == ["c1"]


# ---------------------------------------------------------------------------
# promote_to_golden_set
# ---------------------------------------------------------------------------


def test_promote_builds_golden_items_with_verbatim_selections() -> None:
    candidate = _candidate(
        "c1", QuestionType.FACTUAL_LOOKUP, relevant_chunk_ids=["eu_ai_act:article:1"]
    )

    new_items, result = promote_to_golden_set(
        [candidate], existing_golden=[], author="jane", promotion_date="2026-08-06"
    )

    assert len(new_items) == 1
    item = new_items[0]
    assert item.id == "c1"
    assert item.relevant_chunk_ids == ["eu_ai_act:article:1"]
    assert item.provenance.author == "jane"
    assert item.provenance.date == "2026-08-06"
    assert item.provenance.source_reference == "eu_ai_act:article:1"
    assert result.promoted == ["c1"]


def test_promote_out_of_scope_item_gets_placeholder_source_reference() -> None:
    candidate = _candidate("c1", QuestionType.OUT_OF_SCOPE)

    new_items, _ = promote_to_golden_set(
        [candidate], existing_golden=[], author="jane", promotion_date="2026-08-06"
    )

    assert new_items[0].relevant_chunk_ids == []
    assert "outside" in new_items[0].provenance.source_reference


def test_promote_is_idempotent_skips_ids_already_in_golden_set() -> None:
    candidate = _candidate("c1", QuestionType.FACTUAL_LOOKUP, relevant_chunk_ids=["x"])
    existing = [_golden("c1")]

    new_items, result = promote_to_golden_set(
        [candidate], existing_golden=existing, author="jane", promotion_date="2026-08-06"
    )

    assert new_items == []
    assert result.promoted == []
    assert result.already_in_golden_set == ["c1"]


def test_promote_reports_confirmed_none_for_manual_decision() -> None:
    candidate = _candidate(
        "c1", QuestionType.ROLE_SCOPED, notes=NONE_NOTES
    )

    new_items, result = promote_to_golden_set(
        [candidate], existing_golden=[], author="jane", promotion_date="2026-08-06"
    )

    assert new_items == []
    assert result.needs_manual_decision == [("c1", "question for c1")]


def test_promote_counts_dropped_items() -> None:
    candidate = _candidate("c1", QuestionType.FACTUAL_LOOKUP, notes=DROPPED_NOTES)

    new_items, result = promote_to_golden_set(
        [candidate], existing_golden=[], author="jane", promotion_date="2026-08-06"
    )

    assert new_items == []
    assert result.dropped == 1
