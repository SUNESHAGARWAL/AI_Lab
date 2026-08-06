from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from evals.candidates import CandidateForReview, save_reviewed_candidates
from evals.golden import QuestionType

DROPPED_NOTES = "dropped: weak question, excluded from golden set"
NONE_NOTES = "reviewer confirmed: none of the candidates are relevant"


@dataclass(frozen=True)
class ReviewSummary:
    populated: int
    out_of_scope_empty: int
    confirmed_none: int  # in-scope item, reviewer confirmed none of the candidates fit
    dropped: int
    still_undecided: int


def needs_decision(item: CandidateForReview) -> bool:
    """True iff this item has no real decision recorded yet — including the
    corrupted state this tool exists to fix: verified=True with an empty
    relevant_chunk_ids and no notes on a non-out_of_scope item."""
    if not item.verified:
        return True
    if item.question_type is QuestionType.OUT_OF_SCOPE:
        return False
    if item.relevant_chunk_ids:
        return False
    return item.notes is None


def _parse_indices(text: str, pool_size: int) -> list[int]:
    tokens = [t for t in text.replace(",", " ").split() if t]
    if not tokens:
        raise ValueError("no chunk numbers given")
    indices: list[int] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(f"'{token}' is not a valid chunk number")
        index = int(token)
        if not (1 <= index <= pool_size):
            raise ValueError(f"{index} is out of range (candidates are numbered 1-{pool_size})")
        indices.append(index)
    return indices


def apply_decision(item: CandidateForReview, raw: str) -> CandidateForReview:
    """Parses one line of reviewer input against item.candidate_pool. Raises
    ValueError with a human-readable message on anything unparseable/out-of-range —
    the caller re-prompts on that, never substitutes a default or guesses."""
    text = raw.strip().lower()

    if text == "skip":
        return item.model_copy(
            update={"verified": True, "relevant_chunk_ids": [], "notes": DROPPED_NOTES}
        )

    if text == "none":
        notes = None if item.question_type is QuestionType.OUT_OF_SCOPE else NONE_NOTES
        return item.model_copy(
            update={"verified": True, "relevant_chunk_ids": [], "notes": notes}
        )

    indices = _parse_indices(text, len(item.candidate_pool))
    selected = [item.candidate_pool[i - 1].chunk_id for i in indices]
    return item.model_copy(update={"verified": True, "relevant_chunk_ids": selected})


def _print_item(
    item: CandidateForReview, position: int, total: int, print_fn: Callable[[str], None]
) -> None:
    print_fn(f"\n[{position}/{total}] {item.id}  ({item.difficulty}, {item.question_type})")
    print_fn(item.question)
    print_fn("")
    for i, candidate in enumerate(item.candidate_pool, start=1):
        print_fn(f"  [{i}] {candidate.chunk_id}  (score {candidate.score:.3f})")
        for line in candidate.content.splitlines():
            print_fn(f"      {line}")
        print_fn("")
    print_fn('Select relevant chunk numbers (e.g. "1,3"), "none", or "skip":')


def _summarize(items: list[CandidateForReview]) -> ReviewSummary:
    populated = sum(1 for i in items if i.relevant_chunk_ids)
    out_of_scope_empty = sum(
        1
        for i in items
        if i.question_type is QuestionType.OUT_OF_SCOPE and i.verified and not i.relevant_chunk_ids
    )
    confirmed_none = sum(1 for i in items if i.notes == NONE_NOTES)
    dropped = sum(1 for i in items if i.notes == DROPPED_NOTES)
    still_undecided = sum(1 for i in items if needs_decision(i))
    return ReviewSummary(
        populated=populated,
        out_of_scope_empty=out_of_scope_empty,
        confirmed_none=confirmed_none,
        dropped=dropped,
        still_undecided=still_undecided,
    )


def run_interactive_review(
    items: list[CandidateForReview],
    path: Path,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> ReviewSummary:
    """For each item still needing a decision (in file order): prints it in full
    (question, difficulty, question_type, every candidate_pool entry indexed with
    full content), prompts, applies the decision, and saves the WHOLE list back to
    `path` immediately — nothing is lost if the reviewer stops partway. 'quit'/'q'
    exits the loop early without touching anything not yet decided."""
    items = list(items)
    pending = [i for i, item in enumerate(items) if needs_decision(item)]
    total = len(pending)

    for position, index in enumerate(pending, start=1):
        item = items[index]
        while True:
            _print_item(item, position, total, print_fn)
            raw = input_fn("> ")
            if raw.strip().lower() in ("quit", "q"):
                return _summarize(items)
            try:
                items[index] = apply_decision(item, raw)
            except ValueError as exc:
                print_fn(f"! {exc} — try again.")
                continue
            save_reviewed_candidates(items, path)
            break

    return _summarize(items)
