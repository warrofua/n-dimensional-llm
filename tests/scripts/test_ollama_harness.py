from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_harness(args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.ollama_harness", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_ollama_harness_dry_run_cord_sample():
    output = run_harness(["--dataset", "cord", "--use-sample", "--dry-run"])
    payload = json.loads(output)
    assert payload["dataset"] == "cord"
    assert "prompt" in payload
    assert payload["model"] == "llama3.1:8b"


def test_ollama_harness_dry_run_chartqa_sample():
    output = run_harness(["--dataset", "chartqa", "--use-sample", "--dry-run"])
    payload = json.loads(output)
    assert payload["dataset"] == "chartqa"
    assert "Question" in payload["prompt"]
