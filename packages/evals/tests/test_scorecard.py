from pathlib import Path

import pytest
from evals.golden import Difficulty, GoldenItem, Provenance, QuestionType
from evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from evals.scorecard import aggregate, render_table, write_report


def _item(id_: str, question: str, relevant: list[str], question_type: QuestionType) -> GoldenItem:
    return GoldenItem(
        id=id_,
        question=question,
        relevant_chunk_ids=relevant,
        difficulty=Difficulty.EASY,
        question_type=question_type,
        provenance=Provenance(author="jane", date="2026-08-06", source_reference="test"),
    )


def _fake_retrieve_fn(answers: dict[str, list[str]]):
    async def retrieve_fn(question: str, k: int) -> list[str]:
        return answers[question][:k]

    return retrieve_fn


@pytest.mark.asyncio
async def test_aggregate_computes_metrics_over_scored_items() -> None:
    items = [_item("g1", "q1", ["b"], QuestionType.FACTUAL_LOOKUP)]
    retrieve_fn = _fake_retrieve_fn({"q1": ["a", "b", "c"]})

    scorecard = await aggregate(
        items, retrieve_fn, k_values=[1, 3], golden_set_path="x", retriever_model="m",
        generated_at="2026-08-06T00:00:00Z",
    )

    retrieved, relevant = ["a", "b", "c"], {"b"}
    assert scorecard.total_items == 1
    assert scorecard.scored_items == 1
    assert scorecard.out_of_scope_excluded == 0
    assert scorecard.aggregate_recall_at_k[1] == pytest.approx(recall_at_k(retrieved, relevant, 1))
    assert scorecard.aggregate_recall_at_k[3] == pytest.approx(recall_at_k(retrieved, relevant, 3))
    assert scorecard.aggregate_mrr_at_k[3] == pytest.approx(reciprocal_rank(retrieved, relevant, 3))
    assert scorecard.aggregate_ndcg_at_k[3] == pytest.approx(ndcg_at_k(retrieved, relevant, 3))
    # default behavior (no rerank args passed) must stay unreranked
    assert scorecard.reranked is False
    assert scorecard.reranker_model is None


@pytest.mark.asyncio
async def test_aggregate_excludes_out_of_scope_items_from_metrics() -> None:
    items = [
        _item("g1", "q1", ["a"], QuestionType.FACTUAL_LOOKUP),
        _item("g2", "q2", [], QuestionType.OUT_OF_SCOPE),
    ]
    retrieve_fn = _fake_retrieve_fn({"q1": ["a"], "q2": ["x", "y"]})

    scorecard = await aggregate(
        items, retrieve_fn, k_values=[1], golden_set_path="x", retriever_model="m",
        generated_at="2026-08-06T00:00:00Z",
    )

    assert scorecard.total_items == 2
    assert scorecard.scored_items == 1
    assert scorecard.out_of_scope_excluded == 1
    # only the scored item contributes -> recall@1 must be exactly 1.0, not diluted
    assert scorecard.aggregate_recall_at_k[1] == pytest.approx(1.0)

    oos_score = next(p for p in scorecard.per_item if p.id == "g2")
    assert oos_score.scored is False
    assert oos_score.recall_at_k == {}


@pytest.mark.asyncio
async def test_aggregate_retrieve_fn_never_called_for_out_of_scope_items() -> None:
    calls: list[str] = []

    async def retrieve_fn(question: str, k: int) -> list[str]:
        calls.append(question)
        return []

    items = [_item("g1", "q1", [], QuestionType.OUT_OF_SCOPE)]

    await aggregate(
        items, retrieve_fn, k_values=[1], golden_set_path="x", retriever_model="m",
        generated_at="2026-08-06T00:00:00Z",
    )

    assert calls == []


@pytest.mark.asyncio
async def test_aggregate_handles_zero_golden_items() -> None:
    scorecard = await aggregate(
        [], _fake_retrieve_fn({}), k_values=[1, 3], golden_set_path="x", retriever_model="m",
        generated_at="2026-08-06T00:00:00Z",
    )

    assert scorecard.total_items == 0
    assert scorecard.scored_items == 0
    assert scorecard.aggregate_recall_at_k == {}
    assert scorecard.aggregate_mrr_at_k == {}
    assert scorecard.aggregate_ndcg_at_k == {}


@pytest.mark.asyncio
async def test_render_table_reports_zero_scored_items_cleanly() -> None:
    scorecard = await aggregate(
        [], _fake_retrieve_fn({}), k_values=[1], golden_set_path="x", retriever_model="m",
        generated_at="2026-08-06T00:00:00Z",
    )

    table = render_table(scorecard)

    assert "no scored items" in table.lower()


@pytest.mark.asyncio
async def test_write_report_writes_json_and_markdown(tmp_path: Path) -> None:
    items = [_item("g1", "q1", ["a"], QuestionType.FACTUAL_LOOKUP)]
    scorecard = await aggregate(
        items, _fake_retrieve_fn({"q1": ["a"]}), k_values=[1], golden_set_path="x",
        retriever_model="m", generated_at="2026-08-06T00:00:00Z",
    )

    json_path, md_path = write_report(scorecard, tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    assert "g1" in json_path.read_text(encoding="utf-8")
    assert "recall@k" in md_path.read_text(encoding="utf-8")
