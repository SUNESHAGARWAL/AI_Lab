import re
from collections.abc import Awaitable
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict

from evals.generation_metrics import (
    AbstentionReport,
    appropriate_abstention,
    citation_validity,
    context_precision,
    faithfulness,
)
from evals.generation_metrics import answer_relevancy as _answer_relevancy_metric
from evals.generation_run import GenerationRunResult
from evals.golden import QuestionType
from evals.judge import GatewayJudgeModel
from llm import Gateway
from telemetry import get_logger

logger = get_logger(__name__)


class GenerationPerItemScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question_type: QuestionType
    expected_abstain: bool
    abstained: bool
    abstention_correct: bool
    citation_validity: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None


class GenerationScorecard(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str
    golden_set_path: str
    judge_model: str
    sample_size: int
    total_golden_items: int
    appropriate_abstention: AbstentionReport
    mean_citation_validity: float | None
    mean_faithfulness: float | None
    mean_answer_relevancy: float | None
    mean_context_precision: float | None
    per_item: list[GenerationPerItemScore]


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


async def _safe_judge_call(
    call: Awaitable[float | None], item_id: str, metric_name: str
) -> float | None:
    """A real failure on one item's judge call must not crash the whole scorecard
    run — logged loudly, scored as None (excluded from the metric's mean), never
    silently treated as a passing/failing score. Broad except is deliberate: this
    is the free-tier-flakiness boundary, and the failure modes are real and varied
    — gateway-level (AllProvidersExhausted/BudgetExceeded) AND deepeval's own
    internal parsing errors when a cheap fast-tier judge model returns truncated or
    malformed JSON (confirmed happening in practice against llama-3.1-8b-instant)."""
    try:
        return await call
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning(
            "evals.generation_metric_failed",
            id=item_id,
            metric=metric_name,
            error=type(exc).__name__,
            message=str(exc),
        )
        return None


async def aggregate_generation_scorecard(
    results: list[GenerationRunResult],
    gateway: Gateway,
    judge_model: GatewayJudgeModel,
    golden_set_path: str,
    existing_chunk_ids: set[str],
    generated_at: str,
    total_golden_items: int,
) -> GenerationScorecard:
    abstention = appropriate_abstention(results)
    abstention_by_id = {o.id: o for o in abstention.per_item}

    per_item: list[GenerationPerItemScore] = []
    for result in results:
        logger.info("evals.generation_item_scoring", id=result.id)
        outcome = abstention_by_id[result.id]
        cv = await _safe_judge_call(
            citation_validity(gateway, result, existing_chunk_ids), result.id, "citation_validity"
        )
        faith = await _safe_judge_call(
            faithfulness(judge_model, result), result.id, "faithfulness"
        )
        relevancy = await _safe_judge_call(
            _answer_relevancy_metric(judge_model, result), result.id, "answer_relevancy"
        )
        cp = context_precision(result)
        per_item.append(
            GenerationPerItemScore(
                id=result.id,
                question_type=result.question_type,
                expected_abstain=outcome.expected_abstain,
                abstained=result.abstained,
                abstention_correct=outcome.correct,
                citation_validity=cv,
                faithfulness=faith,
                answer_relevancy=relevancy,
                context_precision=cp,
            )
        )

    return GenerationScorecard(
        generated_at=generated_at,
        golden_set_path=golden_set_path,
        judge_model=judge_model.get_model_name(),
        sample_size=len(results),
        total_golden_items=total_golden_items,
        appropriate_abstention=abstention,
        mean_citation_validity=_mean_or_none([p.citation_validity for p in per_item]),
        mean_faithfulness=_mean_or_none([p.faithfulness for p in per_item]),
        mean_answer_relevancy=_mean_or_none([p.answer_relevancy for p in per_item]),
        mean_context_precision=_mean_or_none([p.context_precision for p in per_item]),
        per_item=per_item,
    )


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def render_generation_table(scorecard: GenerationScorecard) -> str:
    ab = scorecard.appropriate_abstention
    lines = [
        f"Generation (Layer 3) scorecard — {scorecard.generated_at}",
        f"golden set: {scorecard.golden_set_path}",
        f"judge model: {scorecard.judge_model}",
        f"sample size: {scorecard.sample_size} / {scorecard.total_golden_items} golden items",
        "",
        "appropriate_abstention:",
        f"  correct: {ab.correct}/{ab.total}",
        f"  false_abstention_rate (answerable wrongly abstained): "
        f"{ab.false_abstention_rate:.4f}",
        f"  false_answer_rate (out-of-scope wrongly answered): {ab.false_answer_rate:.4f}",
        "",
        f"mean citation_validity: {_fmt(scorecard.mean_citation_validity)}",
        f"mean faithfulness:      {_fmt(scorecard.mean_faithfulness)}",
        f"mean answer_relevancy:  {_fmt(scorecard.mean_answer_relevancy)}",
        f"mean context_precision: {_fmt(scorecard.mean_context_precision)}",
    ]
    return "\n".join(lines)


def write_generation_report(scorecard: GenerationScorecard, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_timestamp = re.sub(r"[^0-9A-Za-z]", "", scorecard.generated_at)
    stem = f"generation_scorecard_{safe_timestamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_generation_table(scorecard), encoding="utf-8")
    return json_path, md_path
