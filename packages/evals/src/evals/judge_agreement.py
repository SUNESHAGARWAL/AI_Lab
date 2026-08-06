"""Human-validation step, deliberately separate from the aggregate scorecard. Run
the judge on a small hand-pickable set of real generation results and print its
verdicts alongside the answer/citations/context so a human can eyeball whether the
judge is calling faithfulness/citation-support correctly before trusting Layer 3's
aggregate numbers at scale."""

from collections.abc import Awaitable

from evals.generation_metrics import answer_relevancy, faithfulness
from evals.generation_run import GenerationRunResult
from evals.judge import CitationSupportVerdict, GatewayJudgeModel, judge_citation_support
from llm import Gateway
from telemetry import get_logger

logger = get_logger(__name__)

_BANNER = (
    "# Judge agreement report — HUMAN VALIDATION STEP\n\n"
    "This is not an aggregate scorecard. Read every case below and confirm the "
    "judge's verdicts actually match your own read of the answer/citations before "
    "trusting `run-generation-eval`'s aggregate numbers. If the judge is marking a "
    "well-cited answer unfaithful, or waving through a citation that doesn't "
    "support its claim, the aggregate Layer 3 numbers are noise until that's fixed."
)


async def _safe[T](call: Awaitable[T], item_id: str, label: str) -> T | None:
    """A real judge-call failure on one case must not abort the whole report —
    logged loudly, rendered as a visible failure marker in the Markdown rather than
    silently dropped. Broad except deliberately: both gateway-level failures and
    deepeval's own internal parsing errors (a cheap judge model returning malformed
    JSON) are real, expected failure modes here — see generation_scorecard.py's
    _safe_judge_call docstring for the same reasoning."""
    try:
        return await call
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning(
            "evals.judge_agreement_call_failed",
            id=item_id,
            label=label,
            error=type(exc).__name__,
            message=str(exc),
        )
        return None


async def build_judge_agreement_report(
    results: list[GenerationRunResult],
    gateway: Gateway,
    judge_model: GatewayJudgeModel,
) -> str:
    sections = [_BANNER]

    for result in results:
        lines = [
            f"\n---\n\n## {result.id}",
            f"**question ({result.question_type.value}):** {result.question}",
            f"**abstained:** {result.abstained}"
            + (f"  (reason: {result.abstain_reason})" if result.abstain_reason else ""),
            f"**answer:** {result.answer or '(none)'}",
        ]

        if result.citations:
            lines.append("\n**citations:**")
            for chunk_id in result.citations:
                chunk_text = result.context_chunks.get(chunk_id, "(not in this run's context)")
                verdict = await _safe(
                    judge_citation_support(gateway, chunk_text, result.answer),
                    result.id,
                    f"citation_support:{chunk_id}",
                )
                if verdict is None:
                    lines.append(f"- `{chunk_id}` — judge call FAILED (see logs)")
                    continue
                assert isinstance(verdict, CitationSupportVerdict)
                lines.append(
                    f"- `{chunk_id}` — judge: supports={verdict.supports} "
                    f"({verdict.reason or 'no reason given'})\n"
                    f"  > {chunk_text[:300]}"
                )
        else:
            lines.append("\n**citations:** (none)")

        if not result.abstained and result.context_chunk_ids:
            faith_score = await _safe(faithfulness(judge_model, result), result.id, "faithfulness")
            lines.append(f"\n**judge faithfulness score:** {faith_score}")
        if not result.abstained:
            relevancy_score = await _safe(
                answer_relevancy(judge_model, result), result.id, "answer_relevancy"
            )
            lines.append(f"**judge answer_relevancy score:** {relevancy_score}")

        expected_abstain = result.question_type.value == "out_of_scope"
        lines.append(
            f"\n**appropriate_abstention:** expected_abstain={expected_abstain}, "
            f"actual_abstained={result.abstained}, "
            f"correct={expected_abstain == result.abstained}"
        )

        sections.append("\n".join(lines))

    return "\n".join(sections)
