import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from evals.golden import Difficulty, GoldenItem, QuestionType
from evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from telemetry import get_logger

logger = get_logger(__name__)

RetrieveFn = Callable[[str, int], Awaitable[list[str]]]


class PerItemScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question_type: QuestionType
    difficulty: Difficulty
    scored: bool
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    reciprocal_rank_at_k: dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)


class Scorecard(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str
    golden_set_path: str
    retriever_model: str
    reranked: bool = False
    reranker_model: str | None = None
    k_values: list[int]
    total_items: int
    scored_items: int
    out_of_scope_excluded: int
    aggregate_recall_at_k: dict[int, float]
    aggregate_mrr_at_k: dict[int, float]
    aggregate_ndcg_at_k: dict[int, float]
    per_item: list[PerItemScore]


def _score_item(item: GoldenItem, retrieved_ids: list[str], k_values: list[int]) -> PerItemScore:
    relevant_ids = set(item.relevant_chunk_ids)
    return PerItemScore(
        id=item.id,
        question_type=item.question_type,
        difficulty=item.difficulty,
        scored=True,
        recall_at_k={k: recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        reciprocal_rank_at_k={k: reciprocal_rank(retrieved_ids, relevant_ids, k) for k in k_values},
        ndcg_at_k={k: ndcg_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
    )


async def aggregate(
    golden_items: list[GoldenItem],
    retrieve_fn: RetrieveFn,
    k_values: list[int],
    golden_set_path: str,
    retriever_model: str,
    generated_at: str,
    reranked: bool = False,
    reranker_model: str | None = None,
) -> Scorecard:
    per_item: list[PerItemScore] = []
    out_of_scope_excluded = 0
    max_k = max(k_values) if k_values else 0

    for item in golden_items:
        is_out_of_scope = item.question_type is QuestionType.OUT_OF_SCOPE
        has_ground_truth = bool(item.relevant_chunk_ids)

        if is_out_of_scope != (not has_ground_truth):
            logger.warning(
                "evals.golden_item_tag_mismatch",
                id=item.id,
                question_type=str(item.question_type),
                relevant_chunk_ids_count=len(item.relevant_chunk_ids),
            )

        if is_out_of_scope or not has_ground_truth:
            out_of_scope_excluded += 1
            per_item.append(
                PerItemScore(
                    id=item.id,
                    question_type=item.question_type,
                    difficulty=item.difficulty,
                    scored=False,
                )
            )
            continue

        retrieved_ids = await retrieve_fn(item.question, max_k)
        per_item.append(_score_item(item, retrieved_ids, k_values))

    scored = [p for p in per_item if p.scored]

    def _agg(field: str) -> dict[int, float]:
        if not scored:
            return {}
        return {k: mean(getattr(p, field)[k] for p in scored) for k in k_values}

    return Scorecard(
        generated_at=generated_at,
        golden_set_path=golden_set_path,
        retriever_model=retriever_model,
        reranked=reranked,
        reranker_model=reranker_model,
        k_values=k_values,
        total_items=len(golden_items),
        scored_items=len(scored),
        out_of_scope_excluded=out_of_scope_excluded,
        aggregate_recall_at_k=_agg("recall_at_k"),
        aggregate_mrr_at_k=_agg("reciprocal_rank_at_k"),
        aggregate_ndcg_at_k=_agg("ndcg_at_k"),
        per_item=per_item,
    )


def render_table(scorecard: Scorecard) -> str:
    reranked_line = f"reranked: {scorecard.reranked}"
    if scorecard.reranked:
        reranked_line += f"  (reranker model: {scorecard.reranker_model})"

    lines = [
        f"Retrieval scorecard — {scorecard.generated_at}",
        f"golden set: {scorecard.golden_set_path}",
        f"retriever model: {scorecard.retriever_model}",
        reranked_line,
        f"total items: {scorecard.total_items}  "
        f"scored: {scorecard.scored_items}  "
        f"out-of-scope excluded: {scorecard.out_of_scope_excluded}",
        "",
    ]

    if scorecard.scored_items == 0:
        lines.append("(no scored items — nothing to aggregate)")
        return "\n".join(lines)

    header = f"{'k':>4} | {'recall@k':>10} | {'MRR@k':>10} | {'nDCG@k':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in scorecard.k_values:
        lines.append(
            f"{k:>4} | "
            f"{scorecard.aggregate_recall_at_k[k]:>10.4f} | "
            f"{scorecard.aggregate_mrr_at_k[k]:>10.4f} | "
            f"{scorecard.aggregate_ndcg_at_k[k]:>10.4f}"
        )
    return "\n".join(lines)


def write_report(scorecard: Scorecard, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_timestamp = re.sub(r"[^0-9A-Za-z]", "", scorecard.generated_at)
    stem = f"retrieval_scorecard_{safe_timestamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_table(scorecard), encoding="utf-8")
    return json_path, md_path
