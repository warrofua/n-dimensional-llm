"""Smoke tests for the command-line scripts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import eval_rd, train


def test_train_script_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "train_history.json"
    result = train.main(
        [
            "--epochs",
            "1",
            "--dataset-size",
            "4",
            "--batch-size",
            "2",
            "--budget",
            "4",
            "--hidden-dim",
            "32",
            "--output",
            str(output_path),
        ]
    )
    assert "history" in result
    history = result["history"]
    assert isinstance(history, list) and history
    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["history"]
    assert "loss" in history[0]


def test_eval_rd_script_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "rd_report.json"
    report = eval_rd.main(
        [
            "--budgets",
            "2",
            "4",
            "--dataset-size",
            "4",
            "--batch-size",
            "2",
            "--hidden-dim",
            "32",
            "--output",
            str(output_path),
        ]
    )
    assert "budgets" in report
    budgets = report["budgets"]
    assert isinstance(budgets, list) and len(budgets) == 2
    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert len(saved["budgets"]) == 2
