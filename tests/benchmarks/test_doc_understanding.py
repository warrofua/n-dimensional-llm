from __future__ import annotations

from benchmarks.doc_understanding import run_benchmark


def test_run_benchmark_smoke() -> None:
    report = run_benchmark(budget_values=(2, 4), dataset_size=6, threshold=400.0, seed=2)

    assert report["dataset_size"] == 6
    assert report["threshold"] == 400.0
    assert len(report["budgets"]) == 2

    for entry in report["budgets"]:
        assert {"budget", "accuracy", "average_kept_tokens", "budget_probe", "retention_probe"} <= set(entry)
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert entry["average_kept_tokens"] >= 0
        assert isinstance(entry["budget_probe"], dict)
        assert isinstance(entry["retention_probe"], dict)
