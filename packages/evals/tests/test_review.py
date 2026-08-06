from pathlib import Path

from evals.candidates import (
    CandidateForReview,
    CandidatePoolItem,
    load_candidates,
    save_reviewed_candidates,
)
from evals.golden import Difficulty, QuestionType
from evals.review import apply_decision, needs_decision, run_interactive_review


def _candidate(
    id_: str,
    question_type: QuestionType,
    *,
    pool_size: int = 2,
    verified: bool = False,
    relevant_chunk_ids: list[str] | None = None,
    notes: str | None = None,
) -> CandidateForReview:
    pool = [
        CandidatePoolItem(chunk_id=f"{id_}-c{i}", score=0.5, content=f"content {i}")
        for i in range(1, pool_size + 1)
    ]
    return CandidateForReview(
        id=id_,
        question=f"question for {id_}",
        difficulty=Difficulty.EASY,
        question_type=question_type,
        candidate_pool=pool,
        verified=verified,
        relevant_chunk_ids=relevant_chunk_ids or [],
        notes=notes,
        generated_at="2026-08-06T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# needs_decision
# ---------------------------------------------------------------------------


def test_needs_decision_true_for_fresh_unverified_item() -> None:
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP)
    assert needs_decision(item) is True


def test_needs_decision_true_for_corrupted_verified_but_empty_state() -> None:
    # exactly the bug this tool exists to fix
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP, verified=True)
    assert needs_decision(item) is True


def test_needs_decision_false_when_selections_recorded() -> None:
    item = _candidate(
        "c1", QuestionType.FACTUAL_LOOKUP, verified=True, relevant_chunk_ids=["c1-c1"]
    )
    assert needs_decision(item) is False


def test_needs_decision_false_for_out_of_scope_verified_empty() -> None:
    item = _candidate("c1", QuestionType.OUT_OF_SCOPE, verified=True)
    assert needs_decision(item) is False


def test_needs_decision_false_when_notes_recorded() -> None:
    item = _candidate(
        "c1", QuestionType.ROLE_SCOPED, verified=True, notes="reviewer confirmed: none"
    )
    assert needs_decision(item) is False


# ---------------------------------------------------------------------------
# apply_decision
# ---------------------------------------------------------------------------


def test_apply_decision_single_index() -> None:
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP, pool_size=3)
    result = apply_decision(item, "2")
    assert result.relevant_chunk_ids == ["c1-c2"]
    assert result.verified is True
    assert result.notes is None


def test_apply_decision_multiple_indices() -> None:
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP, pool_size=3)
    result = apply_decision(item, "1,3")
    assert result.relevant_chunk_ids == ["c1-c1", "c1-c3"]


def test_apply_decision_none_on_out_of_scope_leaves_notes_empty() -> None:
    item = _candidate("c1", QuestionType.OUT_OF_SCOPE)
    result = apply_decision(item, "none")
    assert result.relevant_chunk_ids == []
    assert result.verified is True
    assert result.notes is None


def test_apply_decision_none_on_in_scope_item_sets_notes() -> None:
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP)
    result = apply_decision(item, "none")
    assert result.relevant_chunk_ids == []
    assert result.verified is True
    assert result.notes == "reviewer confirmed: none of the candidates are relevant"


def test_apply_decision_skip_sets_dropped_notes() -> None:
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP)
    result = apply_decision(item, "skip")
    assert result.relevant_chunk_ids == []
    assert result.notes == "dropped: weak question, excluded from golden set"


def test_apply_decision_out_of_range_index_raises() -> None:
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP, pool_size=2)
    try:
        apply_decision(item, "5")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_apply_decision_non_numeric_raises_and_leaves_item_unchanged() -> None:
    item = _candidate("c1", QuestionType.FACTUAL_LOOKUP)
    try:
        apply_decision(item, "abc")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert item.verified is False  # frozen model — original untouched


# ---------------------------------------------------------------------------
# run_interactive_review
# ---------------------------------------------------------------------------


def test_run_interactive_review_saves_after_every_item(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    item1 = _candidate("cand-001", QuestionType.FACTUAL_LOOKUP)
    item2 = _candidate("cand-002", QuestionType.FACTUAL_LOOKUP)
    items = [item1, item2]

    responses = iter(["1", "2"])
    captured: dict[str, list[str]] = {}
    call_count = {"n": 0}

    def input_fn(_prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            # about to prompt for item2 — item1's decision must already be on disk
            saved = load_candidates(path)
            captured["item1_relevant_chunk_ids"] = saved[0].relevant_chunk_ids
        return next(responses)

    summary = run_interactive_review(items, path, input_fn=input_fn, print_fn=lambda _: None)

    assert captured["item1_relevant_chunk_ids"] == ["cand-001-c1"]
    assert summary.populated == 2
    assert summary.still_undecided == 0


def test_run_interactive_review_retries_on_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    items = [_candidate("cand-001", QuestionType.FACTUAL_LOOKUP)]
    responses = iter(["abc", "1"])
    prints: list[str] = []

    summary = run_interactive_review(
        items, path, input_fn=lambda _p: next(responses), print_fn=prints.append
    )

    assert summary.populated == 1
    assert any("try again" in line for line in prints)


def test_run_interactive_review_quit_leaves_remaining_items_undecided(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    items = [
        _candidate("cand-001", QuestionType.FACTUAL_LOOKUP),
        _candidate("cand-002", QuestionType.FACTUAL_LOOKUP),
    ]

    summary = run_interactive_review(
        items, path, input_fn=lambda _p: "quit", print_fn=lambda _: None
    )

    assert summary.still_undecided == 2
    assert summary.populated == 0


def test_run_interactive_review_skips_already_decided_items(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    decided = _candidate(
        "cand-001", QuestionType.OUT_OF_SCOPE, verified=True
    )
    undecided = _candidate("cand-002", QuestionType.FACTUAL_LOOKUP)
    items = [decided, undecided]

    calls = {"n": 0}

    def input_fn(_prompt: str) -> str:
        calls["n"] += 1
        return "1"

    summary = run_interactive_review(items, path, input_fn=input_fn, print_fn=lambda _: None)

    assert calls["n"] == 1  # only the undecided item prompted
    assert summary.out_of_scope_empty == 1
    assert summary.populated == 1


def test_run_interactive_review_summary_categories_cover_every_item(tmp_path: Path) -> None:
    # every decision path exercised at once — regression test for a real bug found
    # during manual review: "none" on an in-scope item wasn't counted in any bucket,
    # so the summary silently didn't add up to the total item count.
    path = tmp_path / "candidates.jsonl"
    items = [
        _candidate("cand-001", QuestionType.FACTUAL_LOOKUP),  # -> populated
        _candidate("cand-002", QuestionType.FACTUAL_LOOKUP),  # -> confirmed_none
        _candidate("cand-003", QuestionType.FACTUAL_LOOKUP),  # -> dropped
        _candidate("cand-004", QuestionType.OUT_OF_SCOPE, verified=True),  # already decided
    ]
    responses = iter(["1", "none", "skip"])

    summary = run_interactive_review(
        items, path, input_fn=lambda _p: next(responses), print_fn=lambda _: None
    )

    total_accounted_for = (
        summary.populated
        + summary.confirmed_none
        + summary.out_of_scope_empty
        + summary.dropped
        + summary.still_undecided
    )
    assert total_accounted_for == len(items)
    assert summary.populated == 1
    assert summary.confirmed_none == 1
    assert summary.dropped == 1
    assert summary.out_of_scope_empty == 1
    assert summary.still_undecided == 0


# ---------------------------------------------------------------------------
# load/save round-trip
# ---------------------------------------------------------------------------


def test_load_and_save_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    items = [
        _candidate("cand-001", QuestionType.FACTUAL_LOOKUP, verified=True, relevant_chunk_ids=["x"])
    ]

    save_reviewed_candidates(items, path)
    loaded = load_candidates(path)

    assert len(loaded) == 1
    assert loaded[0].relevant_chunk_ids == ["x"]
    assert loaded[0].verified is True
