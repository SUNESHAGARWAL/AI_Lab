"""Layer 3 metrics. Deterministic functions (appropriate_abstention,
context_precision, citation_existence) need no gateway/network and are covered by
the fast test suite. Judge-backed functions (citation_validity, faithfulness,
answer_relevancy) make real LLM calls and are exercised manually — see
evals/cli.py's run-generation-eval."""

from pydantic import BaseModel, ConfigDict

from evals.generation_run import GenerationRunResult
from evals.golden import QuestionType
from evals.judge import GatewayJudgeModel, judge_citation_support
from llm import Gateway


class AbstentionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    expected_abstain: bool
    abstained: bool
    correct: bool


class AbstentionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    correct: int
    false_abstention: int
    """Answerable item incorrectly abstained — overly cautious."""
    false_answer: int
    """Out-of-scope item incorrectly answered — the dangerous failure mode."""
    false_abstention_rate: float
    false_answer_rate: float
    per_item: list[AbstentionOutcome]


def appropriate_abstention(results: list[GenerationRunResult]) -> AbstentionReport:
    outcomes: list[AbstentionOutcome] = []
    answerable_total = 0
    out_of_scope_total = 0
    false_abstention = 0
    false_answer = 0

    for result in results:
        expected_abstain = result.question_type is QuestionType.OUT_OF_SCOPE
        correct = expected_abstain == result.abstained
        outcomes.append(
            AbstentionOutcome(
                id=result.id,
                expected_abstain=expected_abstain,
                abstained=result.abstained,
                correct=correct,
            )
        )
        if expected_abstain:
            out_of_scope_total += 1
            if not correct:
                false_answer += 1
        else:
            answerable_total += 1
            if not correct:
                false_abstention += 1

    return AbstentionReport(
        total=len(results),
        correct=sum(1 for o in outcomes if o.correct),
        false_abstention=false_abstention,
        false_answer=false_answer,
        false_abstention_rate=(false_abstention / answerable_total) if answerable_total else 0.0,
        false_answer_rate=(false_answer / out_of_scope_total) if out_of_scope_total else 0.0,
        per_item=outcomes,
    )


def context_precision(result: GenerationRunResult) -> float | None:
    """|context shown to the generator ∩ relevant_chunk_ids| / |context shown|.
    None when nothing was shown (e.g. an out-of-scope short-circuit that never
    reached retrieval) — not a fabricated 0.0/1.0."""
    if not result.context_chunk_ids:
        return None
    relevant = set(result.relevant_chunk_ids)
    hits = sum(1 for cid in result.context_chunk_ids if cid in relevant)
    return hits / len(result.context_chunk_ids)


def citation_existence(citations: list[str], existing_chunk_ids: set[str]) -> float | None:
    """None for an empty citation list (abstained answers) rather than a vacuous 1.0."""
    if not citations:
        return None
    hits = sum(1 for cid in citations if cid in existing_chunk_ids)
    return hits / len(citations)


async def citation_validity(
    gateway: Gateway, result: GenerationRunResult, existing_chunk_ids: set[str]
) -> float | None:
    """Fraction of citations that both (a) exist in the corpus and (b) are judged to
    support the answer. Judged per-citation against the whole answer text — the
    finest grain GeneratedAnswer's flat citations list actually supports; there is
    no per-sentence claim->citation mapping anywhere in this codebase to check
    against."""
    if not result.citations:
        return None
    valid = 0
    for chunk_id in result.citations:
        if chunk_id not in existing_chunk_ids:
            continue
        chunk_text = result.context_chunks.get(chunk_id)
        if chunk_text is None:
            # Cited a real corpus chunk that wasn't part of this run's context —
            # can't judge support without the text; treat as invalid.
            continue
        verdict = await judge_citation_support(gateway, chunk_text, result.answer)
        if verdict.supports:
            valid += 1
    return valid / len(result.citations)


async def faithfulness(judge_model: GatewayJudgeModel, result: GenerationRunResult) -> float | None:
    if result.abstained or not result.context_chunk_ids:
        return None
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=result.question,
        actual_output=result.answer,
        retrieval_context=list(result.context_chunks.values()),
    )
    metric = FaithfulnessMetric(model=judge_model, async_mode=True, include_reason=False)
    return await metric.a_measure(test_case, _show_indicator=False)


async def answer_relevancy(
    judge_model: GatewayJudgeModel, result: GenerationRunResult
) -> float | None:
    if result.abstained:
        return None
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(input=result.question, actual_output=result.answer)
    metric = AnswerRelevancyMetric(model=judge_model, async_mode=True, include_reason=False)
    return await metric.a_measure(test_case, _show_indicator=False)
