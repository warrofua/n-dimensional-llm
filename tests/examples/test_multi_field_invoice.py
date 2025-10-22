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

    fusion = summary["cell_fusion"]
    assert fusion["grid_hw"] == [16, 12]
    assert fusion["feature_dim"] >= 0
    cells = fusion["cells"]
    assert isinstance(cells, list)
    assert len(cells) == 1
    expected_cells = fusion["grid_hw"][0] * fusion["grid_hw"][1]
    assert len(cells[0]) == expected_cells
    if fusion["feature_dim"] > 0 and cells[0]:
        assert all(len(vector) == fusion["feature_dim"] for vector in cells[0])
