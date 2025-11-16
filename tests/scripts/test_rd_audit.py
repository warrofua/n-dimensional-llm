from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_rd_audit_cli_runs_with_sample_dataset():
    output = run(
        [
            sys.executable,
            "-m",
            "scripts.rd_audit",
            "--budgets",
            "4",
            "--dataset-size",
            "1",
            "--use-sample",
        ]
    )
    report = json.loads(output)
    assert report["dataset"] == "CORD-v2"
    assert report["dataset_size"] == 1
    assert "modes" in report
    assert set(report["modes"]) == {"nd", "text"}
