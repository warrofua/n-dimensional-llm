"""Ensure script entry points load with absolute imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> None:
    script_path = Path("scripts") / script_name
    subprocess.run([sys.executable, str(script_path), "--help"], check=True)


def test_train_script_cli() -> None:
    run_script("train.py")


def test_eval_rd_script_cli() -> None:
    run_script("eval_rd.py")
