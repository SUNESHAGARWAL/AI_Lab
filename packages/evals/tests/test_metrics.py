import math

import pytest
from evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_partial_hit_within_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    # top-3 = {a,b,c}; hits = {b} -> 1/3
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_full_hit_at_larger_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    # top-4 = all four; hits = {b,d} -> 2/2 = 1.0
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(1.0)


def test_recall_at_k_no_hits() -> None:
    retrieved = ["a", "c"]
    relevant = {"z"}
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)


def test_recall_at_k_handles_k_larger_than_retrieved_list() -> None:
    retrieved = ["a"]
    relevant = {"a"}
    assert recall_at_k(retrieved, relevant, k=10) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [recall_at_k, reciprocal_rank, ndcg_at_k])
def test_metrics_raise_on_nonpositive_k(fn) -> None:
    with pytest.raises(ValueError):
        fn(["a"], {"a"}, k=0)
    with pytest.raises(ValueError):
        fn(["a"], {"a"}, k=-1)


@pytest.mark.parametrize("fn", [recall_at_k, reciprocal_rank, ndcg_at_k])
def test_metrics_raise_on_empty_relevant_ids(fn) -> None:
    with pytest.raises(ValueError):
        fn(["a", "b"], set(), k=2)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_hit_at_first_position() -> None:
    assert reciprocal_rank(["a", "b"], {"a"}, k=2) == pytest.approx(1.0)


def test_reciprocal_rank_hit_at_second_position() -> None:
    assert reciprocal_rank(["a", "b"], {"b"}, k=2) == pytest.approx(0.5)


def test_reciprocal_rank_no_hit_within_k() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_at_k_perfect_ranking() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    # ideal ranking is exactly what's retrieved -> nDCG = 1.0
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_no_hits() -> None:
    assert ndcg_at_k(["a", "b"], {"z"}, k=2) == pytest.approx(0.0)


def test_ndcg_at_k_out_of_order_hits_hand_computed() -> None:
    # relevant items land at ranks 2 and 4, not the ideal ranks 1 and 2.
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    k = 4

    dcg = 1.0 / math.log2(2 + 1) + 1.0 / math.log2(4 + 1)  # hits at rank 2, rank 4
    idcg = 1.0 / math.log2(1 + 1) + 1.0 / math.log2(2 + 1)  # ideal: both hits first
    expected = dcg / idcg

    assert ndcg_at_k(retrieved, relevant, k) == pytest.approx(expected)
    # sanity: hand-computed value is strictly between 0 and 1 for this case
    assert 0 < expected < 1


def test_ndcg_at_k_more_relevant_items_than_k() -> None:
    # 3 relevant items exist but k=2 — IDCG must be capped at min(k, |relevant|)=2,
    # not len(relevant)=3, otherwise nDCG could never reach 1.0 at this k.
    retrieved = ["a", "b"]
    relevant = {"a", "b", "z"}
    assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)
