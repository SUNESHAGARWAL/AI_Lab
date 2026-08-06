import json
from pathlib import Path

import pytest
from evals.cli import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.integration
def test_run_retrieval_against_empty_golden_set_reports_zero_scored(
    require_postgres: None, tmp_path: Path
) -> None:
    golden_set = tmp_path / "empty_golden.jsonl"
    golden_set.write_text("", encoding="utf-8")
    output_dir = tmp_path / "reports"

    result = runner.invoke(
        app, ["run-retrieval", "--golden-set", str(golden_set), "--output-dir", str(output_dir)]
    )

    assert result.exit_code == 0, result.output
    reports = list(output_dir.glob("retrieval_scorecard_*.json"))
    assert len(reports) == 1
    scorecard = json.loads(reports[0].read_text(encoding="utf-8"))
    assert scorecard["total_items"] == 0
    assert scorecard["scored_items"] == 0


@pytest.mark.integration
def test_run_retrieval_against_real_corpus_produces_sane_scores(
    require_postgres: None, tmp_path: Path
) -> None:
    golden_set = tmp_path / "golden.jsonl"
    items = [
        {
            "schema_version": 1,
            "id": "g-001",
            "question": "What is the subject matter of the AI Act according to Article 1?",
            "relevant_chunk_ids": ["eu_ai_act:article:1"],
            "difficulty": "easy",
            "question_type": "factual_lookup",
            "provenance": {
                "author": "integration-test",
                "date": "2026-08-06",
                "source_reference": "eu_ai_act:article:1",
            },
        },
        {
            "schema_version": 1,
            "id": "g-002",
            "question": "What is the right to erasure under Article 17 of the GDPR?",
            "relevant_chunk_ids": ["gdpr:article:17"],
            "difficulty": "easy",
            "question_type": "factual_lookup",
            "provenance": {
                "author": "integration-test",
                "date": "2026-08-06",
                "source_reference": "gdpr:article:17",
            },
        },
    ]
    golden_set.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    output_dir = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "run-retrieval",
            "--golden-set",
            str(golden_set),
            "--output-dir",
            str(output_dir),
            "--k",
            "1,5",
        ],
    )

    assert result.exit_code == 0, result.output
    reports = list(output_dir.glob("retrieval_scorecard_*.json"))
    scorecard = json.loads(reports[0].read_text(encoding="utf-8"))

    assert scorecard["scored_items"] == 2
    # both questions are direct single-provision lookups against ingested real
    # content — the retriever should find the exact chunk within the top 5.
    assert scorecard["aggregate_recall_at_k"]["5"] == pytest.approx(1.0)
