from __future__ import annotations

from benchmarks import run_long_qa_benchmark, run_video_qa_benchmark


def _assert_budget_payload(entry: dict) -> None:
    assert "metrics" in entry
    metrics = entry["metrics"]
    assert "encoder_latency_seconds" in metrics
    assert "fano_error_bound" in metrics
    assert "registration_post_distortion" in metrics
    assert "ablations" in entry
    assert isinstance(entry["ablations"], dict)
    if entry["ablations"]:
        for payload in entry["ablations"].values():
            assert "accuracy" in payload
            assert "metrics" in payload


def test_long_qa_benchmark_smoke() -> None:
    report = run_long_qa_benchmark(
        budget_values=(5,), dataset_size=4, seed=1, num_turns=4
    )

    assert report["dataset"] == "synthetic-long-qa"
    assert report["dataset_size"] == 4
    assert len(report["budgets"]) == 1

    entry = report["budgets"][0]
    assert entry["budget"] == 5
    _assert_budget_payload(entry)


def test_video_qa_benchmark_smoke() -> None:
    report = run_video_qa_benchmark(
        budget_values=(6,), dataset_size=3, seed=2, num_frames=5
    )

    assert report["dataset"] == "synthetic-video-qa"
    assert report["dataset_size"] == 3
    assert len(report["budgets"]) == 1

    entry = report["budgets"][0]
    assert entry["budget"] == 6
    _assert_budget_payload(entry)
