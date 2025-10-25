from __future__ import annotations

from benchmarks.doc_understanding import (
    run_benchmark,
    run_doclaynet_benchmark,
    run_funsd_benchmark,
)


def test_run_benchmark_smoke() -> None:
    report = run_benchmark(budget_values=(2, 4), dataset_size=6, threshold=400.0, seed=2)

    assert report["dataset_size"] == 6
    assert report["threshold"] == 400.0
    assert len(report["budgets"]) == 2

    for entry in report["budgets"]:
        assert {
            "budget",
            "accuracy",
            "average_kept_tokens",
            "budget_probe",
            "retention_probe",
            "metrics",
        } <= set(entry)
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert entry["average_kept_tokens"] >= 0
        assert isinstance(entry["budget_probe"], dict)
        assert isinstance(entry["retention_probe"], dict)
        assert isinstance(entry["metrics"], dict)
        assert "cell_fusions" in entry
        assert isinstance(entry["cell_fusions"], list)
        assert "ablations" in entry
        assert isinstance(entry["ablations"], dict)
        assert "encoder_latency_seconds" in entry["metrics"]
        assert "fano_error_bound" in entry["metrics"]
        assert "registration_pre_distortion" in entry["metrics"]
        if entry["ablations"]:
            for ablation in entry["ablations"].values():
                assert "metrics" in ablation
                assert "accuracy" in ablation


def test_run_funsd_benchmark_smoke() -> None:
    report = run_funsd_benchmark(budget_values=(6,), dataset_size=2, use_sample=True)

    assert report["dataset"] == "FUNSD"
    assert report["dataset_size"] == 2
    assert report["use_sample"] is True
    assert len(report["budgets"]) == 1

    entry = report["budgets"][0]
    assert entry["budget"] == 6
    assert 0.0 <= entry["accuracy"] <= 1.0
    assert isinstance(entry["metrics"], dict)
    assert "cell_fusions" in entry
    assert entry["cell_fusions"]
    assert "ablations" in entry
    assert isinstance(entry["ablations"], dict)
    assert "drop_layout" in entry["ablations"]
    drop_layout_accuracy = entry["ablations"]["drop_layout"]["accuracy"]
    assert drop_layout_accuracy <= entry["accuracy"]
    assert "drop_text" in entry["ablations"]
    drop_text_accuracy = entry["ablations"]["drop_text"]["accuracy"]
    assert drop_text_accuracy <= entry["accuracy"]


def test_run_doclaynet_benchmark_smoke() -> None:
    report = run_doclaynet_benchmark(budget_values=(4,), dataset_size=2, use_sample=True)

    assert report["dataset"] == "DocLayNet"
    assert report["dataset_size"] == 2
    assert report["use_sample"] is True
    assert len(report["budgets"]) == 1

    entry = report["budgets"][0]
    assert entry["budget"] == 4
    assert 0.0 <= entry["accuracy"] <= 1.0
    assert isinstance(entry["metrics"], dict)
    assert "encoder_latency_seconds" in entry["metrics"]
    assert "cell_fusions" in entry
    assert entry["cell_fusions"]
    assert "ablations" in entry
    assert isinstance(entry["ablations"], dict)
