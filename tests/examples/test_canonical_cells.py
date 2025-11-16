from __future__ import annotations

from typing import Any, Sequence

import pytest

from nd_llm.utils import (
    DEFAULT_BACKEND,
    NUMPY_AVAILABLE,
    TORCH_AVAILABLE,
    aggregate_fields,
    assign_to_cells,
    rasterize_cells,
)

try:  # pragma: no cover - optional dependency for backend conversions
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for backend conversions
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _to_backend(data: Sequence[Sequence[float]], backend: str) -> Any:
    if backend == "torch" and TORCH_AVAILABLE and torch is not None:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor
    if backend == "numpy" and NUMPY_AVAILABLE and _np is not None:
        array = _np.asarray(data, dtype=_np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        return array
    nested = [[float(value) for value in row] for row in data]
    return [nested]


def _to_list(value: Any) -> Any:
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if _np is not None and isinstance(value, _np.ndarray):
        return value.tolist()
    return value


def test_rasterize_cells_shape_consistency() -> None:
    backend = "numpy" if NUMPY_AVAILABLE else "python"
    centers = rasterize_cells(2, grid_hw=(4, 3), backend=backend)
    if backend == "numpy":
        assert centers.shape == (2, 12, 2)
    else:
        assert isinstance(centers, list)
        assert len(centers) == 2
        assert len(centers[0]) == 12


def test_assign_to_cells_weights_are_normalised() -> None:
    backend = (
        DEFAULT_BACKEND if DEFAULT_BACKEND != "torch" or TORCH_AVAILABLE else "numpy"
    )
    centers = rasterize_cells(1, grid_hw=(1, 2), backend=backend)
    coords = _to_backend([[0.0, 0.0], [0.0, 1.0]], backend)
    weights = assign_to_cells(coords, centers, tau=1e-4, backend=backend)
    weights_list = _to_list(weights)
    assert len(weights_list[0]) == 2
    left_weights = weights_list[0][0]
    right_weights = weights_list[0][1]
    assert pytest.approx(sum(left_weights), rel=1e-6, abs=1e-6) == 1.0
    assert pytest.approx(sum(right_weights), rel=1e-6, abs=1e-6) == 1.0
    assert left_weights[0] > 0.999
    assert right_weights[1] > 0.999


def test_aggregate_fields_mean_is_deterministic() -> None:
    backend = "numpy" if NUMPY_AVAILABLE else DEFAULT_BACKEND
    centers = rasterize_cells(1, grid_hw=(1, 2), backend=backend)
    field_one = {
        "tokens": _to_backend([[1.0], [3.0]], backend),
        "coords": _to_backend([[0.0, 0.0], [0.0, 1.0]], backend),
    }
    field_two = {
        "tokens": _to_backend([[2.0], [4.0]], backend),
        "coords": _to_backend([[0.0, 0.0], [0.0, 1.0]], backend),
    }
    fused_a = aggregate_fields(
        [field_one, field_two], centers, agg="mean", tau=1e-4, backend=backend
    )
    fused_b = aggregate_fields(
        [field_one, field_two], centers, agg="mean", tau=1e-4, backend=backend
    )
    fused_a_list = _to_list(fused_a)
    fused_b_list = _to_list(fused_b)
    for row_a, row_b in zip(fused_a_list[0], fused_b_list[0]):
        assert pytest.approx(row_a[0], rel=1e-6) == row_b[0]
    assert pytest.approx(fused_a_list[0][0][0], rel=1e-6) == 1.5
    assert pytest.approx(fused_a_list[0][1][0], rel=1e-6) == 3.5
