from __future__ import annotations

from benchmarks.chartqa import (
    build_chartqa_encoders,
    build_chartqa_registry,
    chartqa_fields,
    chartqa_answer,
    load_chartqa_dataset,
    run_chartqa_benchmark,
)


def test_chartqa_sample_fields() -> None:
    dataset = load_chartqa_dataset(use_sample=True, limit=1)
    assert dataset
    registry = build_chartqa_registry()
    build_chartqa_encoders(registry)

    doc = dataset[0]
    fields = chartqa_fields(doc)
    assert set(fields) == {"question", "chart"}
    assert fields["question"]
    assert fields["chart"]
    assert chartqa_answer(doc)


def test_run_chartqa_benchmark_smoke() -> None:
    report = run_chartqa_benchmark(budget_values=(2,), dataset_size=1, use_sample=True)
    assert report["dataset"] == "ChartQA"
    assert len(report["budgets"]) == 1
    entry = report["budgets"][0]
    assert entry["budget"] == 2
    assert 0.0 <= entry["accuracy"] <= 1.0
