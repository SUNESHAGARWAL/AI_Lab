import pytest
from evals.generation_metrics import appropriate_abstention, citation_existence, context_precision
from evals.generation_run import GenerationRunResult
from evals.golden import Difficulty, QuestionType


def _result(
    id_: str = "g1",
    question_type: QuestionType = QuestionType.FACTUAL_LOOKUP,
    abstained: bool = False,
    relevant_chunk_ids: list[str] | None = None,
    context_chunk_ids: list[str] | None = None,
    citations: list[str] | None = None,
) -> GenerationRunResult:
    return GenerationRunResult(
        id=id_,
        question="q",
        question_type=question_type,
        difficulty=Difficulty.EASY,
        relevant_chunk_ids=relevant_chunk_ids or [],
        answer="a",
        citations=citations or [],
        confidence=0.9,
        abstained=abstained,
        abstain_reason=None,
        context_chunk_ids=context_chunk_ids or [],
        context_chunks={},
    )


def test_appropriate_abstention_out_of_scope_correctly_abstained() -> None:
    results = [_result(question_type=QuestionType.OUT_OF_SCOPE, abstained=True)]
    report = appropriate_abstention(results)
    assert report.correct == 1
    assert report.false_answer == 0
    assert report.false_abstention == 0
    assert report.false_answer_rate == 0.0


def test_appropriate_abstention_out_of_scope_wrongly_answered() -> None:
    results = [_result(question_type=QuestionType.OUT_OF_SCOPE, abstained=False)]
    report = appropriate_abstention(results)
    assert report.correct == 0
    assert report.false_answer == 1
    assert report.false_abstention == 0
    assert report.false_answer_rate == 1.0


def test_appropriate_abstention_answerable_correctly_answered() -> None:
    results = [_result(question_type=QuestionType.FACTUAL_LOOKUP, abstained=False)]
    report = appropriate_abstention(results)
    assert report.correct == 1
    assert report.false_abstention == 0
    assert report.false_answer == 0
    assert report.false_abstention_rate == 0.0


def test_appropriate_abstention_answerable_wrongly_abstained() -> None:
    results = [_result(question_type=QuestionType.FACTUAL_LOOKUP, abstained=True)]
    report = appropriate_abstention(results)
    assert report.correct == 0
    assert report.false_abstention == 1
    assert report.false_answer == 0
    assert report.false_abstention_rate == 1.0


def test_appropriate_abstention_rates_reported_separately() -> None:
    results = [
        # false_abstention: answerable item wrongly abstained
        _result(id_="a", question_type=QuestionType.FACTUAL_LOOKUP, abstained=True),
        # false_answer: out-of-scope item wrongly answered
        _result(id_="b", question_type=QuestionType.OUT_OF_SCOPE, abstained=False),
        _result(id_="c", question_type=QuestionType.FACTUAL_LOOKUP, abstained=False),  # correct
        _result(id_="d", question_type=QuestionType.OUT_OF_SCOPE, abstained=True),  # correct
    ]
    report = appropriate_abstention(results)
    assert report.total == 4
    assert report.correct == 2
    assert report.false_abstention == 1
    assert report.false_answer == 1
    assert report.false_abstention_rate == pytest.approx(0.5)  # 1 of 2 answerable items
    assert report.false_answer_rate == pytest.approx(0.5)  # 1 of 2 out-of-scope items


def test_context_precision_exact_fraction() -> None:
    result = _result(
        relevant_chunk_ids=["a", "b"], context_chunk_ids=["a", "c", "d"]
    )
    assert context_precision(result) == pytest.approx(1 / 3)


def test_context_precision_none_when_no_context_shown() -> None:
    result = _result(context_chunk_ids=[])
    assert context_precision(result) is None


def test_citation_existence_exact_fraction() -> None:
    existing = {"a", "b"}
    assert citation_existence(["a", "b", "c"], existing) == pytest.approx(2 / 3)


def test_citation_existence_none_when_no_citations() -> None:
    assert citation_existence([], {"a"}) is None


def test_citation_existence_all_present() -> None:
    assert citation_existence(["a", "b"], {"a", "b"}) == pytest.approx(1.0)


def test_citation_existence_none_present() -> None:
    assert citation_existence(["z"], {"a", "b"}) == pytest.approx(0.0)
