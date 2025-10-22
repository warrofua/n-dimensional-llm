from __future__ import annotations

from pathlib import Path

from examples.multi_field_invoice import run_demo


def test_multi_field_invoice_example(tmp_path: Path) -> None:
    summary = run_demo(storage_dir=tmp_path)

    compression = summary["compression"]
    assert compression.metrics["information_bound"] >= 0.0
    assert summary["stm_key"] in summary["usage_log"]

    stored_tensor = summary["stm_entry"]["tensor"]
    assert isinstance(stored_tensor, list)
    flat = stored_tensor[0] if stored_tensor and isinstance(stored_tensor[0], list) else stored_tensor
    assert all(isinstance(value, float) for value in flat)

    retention = summary["retention_probe"]
    assert retention["total_retained"] >= 1
    assert summary["stm_key"] in retention["sampled_keys"]
