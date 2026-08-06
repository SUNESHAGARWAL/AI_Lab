from dataclasses import dataclass

from evals.candidates import CandidateForReview
from evals.golden import GoldenItem, Provenance, QuestionType
from evals.review import DROPPED_NOTES, NONE_NOTES

_OUT_OF_SCOPE_SOURCE = "n/a — outside AI Act/GDPR subject matter"


@dataclass(frozen=True)
class PromotionResult:
    promoted: list[str]
    already_in_golden_set: list[str]
    needs_manual_decision: list[tuple[str, str]]
    dropped: int


def _source_reference(item: CandidateForReview) -> str:
    if item.relevant_chunk_ids:
        return ",".join(item.relevant_chunk_ids)
    return _OUT_OF_SCOPE_SOURCE


def promotable_candidates(
    candidates: list[CandidateForReview],
) -> tuple[list[CandidateForReview], list[CandidateForReview]]:
    """Splits reviewed candidates into (promotable, needs_manual_decision).
    Promotable = a real selection was recorded, or the item is a decided
    out_of_scope item. Confirmed-none in-scope items (question_type != out_of_scope
    but the reviewer confirmed no candidate fits) are never silently promoted —
    promoting them with an empty relevant_chunk_ids would make them functionally
    indistinguishable from out_of_scope in the scorecard, which conflates "correctly
    outside the corpus" with "retrieval currently fails on this in-scope question."
    They're returned separately for a human to resolve by hand. Dropped
    (weak-question) items are silently excluded from both lists — nothing to
    promote, nothing to flag."""
    promotable: list[CandidateForReview] = []
    needs_manual: list[CandidateForReview] = []
    for item in candidates:
        if item.relevant_chunk_ids or (
            item.question_type is QuestionType.OUT_OF_SCOPE and item.verified
        ):
            promotable.append(item)
        elif item.notes == NONE_NOTES:
            needs_manual.append(item)
        # item.notes == DROPPED_NOTES, or still undecided: excluded from both lists
    return promotable, needs_manual


def promote_to_golden_set(
    candidates: list[CandidateForReview],
    existing_golden: list[GoldenItem],
    author: str,
    promotion_date: str,
) -> tuple[list[GoldenItem], PromotionResult]:
    """Builds new GoldenItems from already-reviewed candidates — every
    relevant_chunk_ids value is copied verbatim from what the human already
    selected in review-candidates, nothing is invented here. Idempotent: ids
    already present in existing_golden are skipped, not duplicated."""
    existing_ids = {g.id for g in existing_golden}
    promotable, needs_manual = promotable_candidates(candidates)

    new_items: list[GoldenItem] = []
    promoted_ids: list[str] = []
    skipped_ids: list[str] = []

    for item in promotable:
        if item.id in existing_ids:
            skipped_ids.append(item.id)
            continue
        new_items.append(
            GoldenItem(
                id=item.id,
                question=item.question,
                relevant_chunk_ids=item.relevant_chunk_ids,
                difficulty=item.difficulty,
                question_type=item.question_type,
                provenance=Provenance(
                    author=author,
                    date=promotion_date,
                    source_reference=_source_reference(item),
                ),
            )
        )
        promoted_ids.append(item.id)

    dropped_count = sum(1 for c in candidates if c.notes == DROPPED_NOTES)

    result = PromotionResult(
        promoted=promoted_ids,
        already_in_golden_set=skipped_ids,
        needs_manual_decision=[(i.id, i.question) for i in needs_manual],
        dropped=dropped_count,
    )
    return new_items, result
