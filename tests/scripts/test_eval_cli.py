"""Integration test for the eval_rd CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_eval_cli_runs_without_name_error() -> None:
    script_path = Path("scripts") / "eval_rd.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--budgets",
            "4",
            "--dataset-size",
            "4",
        ],
        check=True,
    )
