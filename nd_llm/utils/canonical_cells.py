"""Canonical cell rasterisation and aggregation utilities."""

from __future__ import annotations

import math
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    TYPE_CHECKING,
    TypeAlias,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from numpy.typing import NDArray

    NumpyArray: TypeAlias = NDArray[Any]
else:
    NumpyArray: TypeAlias = Any

try:  # pragma: no cover - numpy is optional in some environments
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - fallback when numpy is missing
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - torch is optional during testing
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None  # type: ignore[assignment]

_NUMPY_AVAILABLE = _np is not None
_TORCH_AVAILABLE = torch is not None

Backend = str

TORCH_AVAILABLE: bool = _TORCH_AVAILABLE
NUMPY_AVAILABLE: bool = _NUMPY_AVAILABLE
DEFAULT_BACKEND: Backend = (
    "torch" if TORCH_AVAILABLE else "numpy" if NUMPY_AVAILABLE else "python"
)


def rasterize_cells(
    batch_size: int,
    grid_hw: tuple[int, int] = (32, 24),
    *,
    backend: Backend | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Return canonical cell centre coordinates for a regular grid."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    height, width = (int(grid_hw[0]), int(grid_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive integers")

    resolved_backend = _resolve_backend(backend)

    if resolved_backend == "torch":
        if torch is None:  # pragma: no cover - defensive
            raise RuntimeError("torch backend requested but torch is unavailable")
        tensor_dtype = dtype if dtype is not None else torch.float32
        tensor_device = device if device is not None else torch.device("cpu")
        ys = torch.linspace(0.0, 1.0, steps=height, device=tensor_device, dtype=tensor_dtype)
        xs = torch.linspace(0.0, 1.0, steps=width, device=tensor_device, dtype=tensor_dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        centres_tensor = torch.stack([grid_y, grid_x], dim=-1).view(1, height * width, 2)
        return centres_tensor.repeat(batch_size, 1, 1)

    if resolved_backend == "numpy":
        if not NUMPY_AVAILABLE:  # pragma: no cover - defensive
            raise RuntimeError("numpy backend requested but numpy is unavailable")
        array_dtype = dtype if dtype is not None else _np.float32
        ys = _np.linspace(0.0, 1.0, num=height, dtype=array_dtype)
        xs = _np.linspace(0.0, 1.0, num=width, dtype=array_dtype)
        grid_y, grid_x = _np.meshgrid(ys, xs, indexing="ij")
        centres_array = _np.stack([grid_y, grid_x], axis=-1).reshape(1, height * width, 2)
        if batch_size == 1:
            return centres_array
        return _np.repeat(centres_array, repeats=batch_size, axis=0)

    # Python fallback for environments without numpy/torch
    ys = _linspace_python(height)
    xs = _linspace_python(width)
    centres_list: List[List[List[float]]] = []
    for _ in range(batch_size):
        batch_centres: List[List[float]] = []
        for row in range(height):
            for col in range(width):
                batch_centres.append([ys[row], xs[col]])
        centres_list.append(batch_centres)
    return centres_list


def assign_to_cells(
    coords: Any,
    cell_centers: Any,
    *,
    tau: float = 0.1,
    backend: Backend | None = None,
) -> Any:
    """Return soft cell assignment weights for the provided coordinates."""

    if tau <= 0:
        raise ValueError("tau must be strictly positive")
    resolved_backend = _resolve_backend(backend, coords, cell_centers)

    if resolved_backend == "torch":
        return _assign_torch(coords, cell_centers, tau)
    if resolved_backend == "numpy":
        return _assign_numpy(coords, cell_centers, tau)
    return _assign_python(coords, cell_centers, tau)


def aggregate_fields(
    field_tokens: Sequence[Mapping[str, Any]],
    cell_centers: Any,
    *,
    agg: str = "mean",
    tau: float = 0.1,
    backend: Backend | None = None,
) -> Any:
    """Aggregate field embeddings into canonical cells."""

    if agg not in {"mean", "sum"}:
        raise ValueError("agg must be either 'mean' or 'sum'")
    resolved_backend = _resolve_backend(
        backend,
        cell_centers,
        *[entry.get("tokens") for entry in field_tokens],
    )

    if resolved_backend == "torch":
        return _aggregate_torch(field_tokens, cell_centers, agg, tau)
    if resolved_backend == "numpy":
        return _aggregate_numpy(field_tokens, cell_centers, agg, tau)
    return _aggregate_python(field_tokens, cell_centers, agg, tau)


def _resolve_backend(backend: Backend | None, *samples: Any) -> Backend:
    if backend is not None:
        lowered = backend.lower()
        if lowered == "torch" and not TORCH_AVAILABLE:
            raise ValueError("torch backend requested but torch is unavailable")
        if lowered == "numpy" and not NUMPY_AVAILABLE:
            raise ValueError("numpy backend requested but numpy is unavailable")
        if lowered not in {"torch", "numpy", "python"}:
            raise ValueError(f"Unsupported backend '{backend}'")
        return lowered

    for sample in samples:
        if TORCH_AVAILABLE and torch is not None and torch.is_tensor(sample):
            return "torch"
    for sample in samples:
        if NUMPY_AVAILABLE and _np is not None and isinstance(sample, _np.ndarray):
            return "numpy"
    return DEFAULT_BACKEND


def _linspace_python(length: int) -> List[float]:
    if length == 1:
        return [0.0]
    step = 1.0 / float(length - 1)
    return [index * step for index in range(length)]


def _assign_torch(coords: Any, cell_centers: Any, tau: float) -> Any:
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("torch backend requested but torch is unavailable")
    coords_tensor = torch.as_tensor(coords, dtype=torch.float32)
    centres_tensor = torch.as_tensor(cell_centers, dtype=coords_tensor.dtype, device=coords_tensor.device)
    if coords_tensor.ndim != 3:
        raise ValueError("coords must be a 3D tensor with shape (B, N, K)")
    if centres_tensor.ndim != 3:
        raise ValueError("cell_centers must be a 3D tensor with shape (B, C, K)")
    if coords_tensor.size(0) != centres_tensor.size(0):
        raise ValueError("coords and cell_centers must share the same batch dimension")
    if coords_tensor.size(-1) != centres_tensor.size(-1):
        raise ValueError("coords and cell_centers must share the same coordinate dimension")
    diff = coords_tensor.unsqueeze(2) - centres_tensor.unsqueeze(1)
    dist2 = diff.pow(2).sum(dim=-1)
    scaled = -dist2 / (2.0 * tau * tau)
    return torch.softmax(scaled, dim=-1)


def _assign_numpy(coords: Any, cell_centers: Any, tau: float) -> Any:
    if not NUMPY_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError("numpy backend requested but numpy is unavailable")
    coords_array = _np.asarray(coords, dtype=_np.float32)
    centres_array = _np.asarray(cell_centers, dtype=coords_array.dtype)
    if coords_array.ndim != 3:
        raise ValueError("coords must have shape (B, N, K)")
    if centres_array.ndim != 3:
        raise ValueError("cell_centers must have shape (B, C, K)")
    if coords_array.shape[0] != centres_array.shape[0]:
        raise ValueError("coords and cell_centers must share the batch dimension")
    if coords_array.shape[2] != centres_array.shape[2]:
        raise ValueError("coords and cell_centers must share the coordinate dimension")
    diff = coords_array[:, :, None, :] - centres_array[:, None, :, :]
    dist2 = (diff**2).sum(axis=-1)
    scaled = -dist2 / (2.0 * tau * tau)
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = _np.exp(scaled)
    denom = exp.sum(axis=-1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp / denom


def _assign_python(coords: Any, cell_centers: Any, tau: float) -> List[List[List[float]]]:
    coords_list = _materialise_list(coords)
    centres_list = _materialise_list(cell_centers)
    if not coords_list:
        return [[[]]]
    batch_size = len(coords_list)
    weights: List[List[List[float]]] = []
    for batch_index in range(batch_size):
        coord_batch = coords_list[batch_index]
        centre_batch = centres_list[min(batch_index, len(centres_list) - 1)]
        batch_weights: List[List[float]] = []
        for coord in coord_batch:
            logits: List[float] = []
            for centre in centre_batch:
                dist2 = sum((float(a) - float(b)) ** 2 for a, b in zip(coord, centre))
                logits.append(-dist2 / (2.0 * tau * tau))
            if logits:
                max_logit = max(logits)
                exp_values = [math.exp(value - max_logit) for value in logits]
                total = sum(exp_values) or 1.0
                batch_weights.append([value / total for value in exp_values])
            else:
                batch_weights.append([])
        weights.append(batch_weights)
    return weights


def _aggregate_torch(
    field_tokens: Sequence[Mapping[str, Any]],
    cell_centers: Any,
    agg: str,
    tau: float,
) -> Any:
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("torch backend requested but torch is unavailable")
    centres = torch.as_tensor(cell_centers, dtype=torch.float32)
    if centres.ndim != 3:
        raise ValueError("cell_centers must have shape (B, C, K)")
    batch, cells, coord_dim = centres.shape
    fused: torch.Tensor | None = None
    totals: torch.Tensor | None = None
    feature_dim: int | None = None

    for entry in field_tokens:
        tokens = torch.as_tensor(entry.get("tokens"), dtype=torch.float32)
        coords = torch.as_tensor(entry.get("coords"), dtype=torch.float32)
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (B, N, D)")
        if coords.ndim != 3:
            raise ValueError("coords must have shape (B, N, K)")
        if tokens.size(0) != batch or coords.size(0) != batch:
            raise ValueError("tokens/coords must match cell batch dimension")
        if coords.size(2) != coord_dim:
            raise ValueError("coords must match cell coordinate dimensionality")
        if feature_dim is None:
            feature_dim = tokens.size(2)
            fused = torch.zeros(batch, cells, feature_dim, dtype=tokens.dtype)
            totals = torch.zeros(batch, cells, dtype=tokens.dtype)
        elif tokens.size(2) != feature_dim:
            raise ValueError("all token tensors must share the same embedding dimension")

        weights = assign_to_cells(coords, centres, tau=tau, backend="torch")
        fused = fused + torch.einsum("bnc,bnd->bcd", weights, tokens)
        totals = totals + weights.sum(dim=1)

    if fused is None or totals is None:
        return torch.zeros(batch, cells, 0, dtype=centres.dtype)

    if agg == "mean":
        fused = fused / totals.clamp_min(1e-9).unsqueeze(-1)
    return fused


def _aggregate_numpy(
    field_tokens: Sequence[Mapping[str, Any]],
    cell_centers: Any,
    agg: str,
    tau: float,
) -> Any:
    if not NUMPY_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError("numpy backend requested but numpy is unavailable")
    centres = _np.asarray(cell_centers, dtype=_np.float32)
    if centres.ndim != 3:
        raise ValueError("cell_centers must have shape (B, C, K)")
    batch, cells, coord_dim = centres.shape
    fused: NumpyArray | None = None
    totals: NumpyArray | None = None
    feature_dim: int | None = None

    for entry in field_tokens:
        tokens = _np.asarray(entry.get("tokens"), dtype=_np.float32)
        coords = _np.asarray(entry.get("coords"), dtype=_np.float32)
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (B, N, D)")
        if coords.ndim != 3:
            raise ValueError("coords must have shape (B, N, K)")
        if tokens.shape[0] != batch or coords.shape[0] != batch:
            raise ValueError("tokens/coords must match cell batch dimension")
        if coords.shape[2] != coord_dim:
            raise ValueError("coords must match cell coordinate dimensionality")
        if feature_dim is None:
            feature_dim = int(tokens.shape[2])
            fused = _np.zeros((batch, cells, feature_dim), dtype=tokens.dtype)
            totals = _np.zeros((batch, cells), dtype=tokens.dtype)
        elif int(tokens.shape[2]) != feature_dim:
            raise ValueError("all token tensors must share the same embedding dimension")

        weights = assign_to_cells(coords, centres, tau=tau, backend="numpy")
        fused = fused + _np.einsum("bnc,bnd->bcd", weights, tokens)
        totals = totals + weights.sum(axis=1)

    if fused is None or totals is None:
        return _np.zeros((batch, cells, 0), dtype=centres.dtype)

    if agg == "mean":
        fused = fused / _np.clip(totals[..., None], a_min=1e-9, a_max=None)
    return fused


def _aggregate_python(
    field_tokens: Sequence[Mapping[str, Any]],
    cell_centers: Any,
    agg: str,
    tau: float,
) -> List[List[List[float]]]:
    centres = _materialise_list(cell_centers)
    if not centres:
        return []
    batch = len(centres)
    cells = len(centres[0]) if centres[0] else 0
    fused: List[List[List[float]]] | None = None
    totals: List[List[float]] | None = None
    feature_dim: int | None = None

    for entry in field_tokens:
        tokens = _materialise_list(entry.get("tokens"))
        coords = _materialise_list(entry.get("coords"))
        if len(tokens) != batch or len(coords) != batch:
            raise ValueError("tokens/coords must match cell batch dimension")
        if not tokens:
            continue
        if feature_dim is None:
            feature_dim = len(tokens[0][0]) if tokens[0] else 0
            fused = [
                [[0.0 for _ in range(feature_dim)] for _ in range(cells)]
                for _ in range(batch)
            ]
            totals = [[0.0 for _ in range(cells)] for _ in range(batch)]
        elif tokens and len(tokens[0][0]) != feature_dim:
            raise ValueError("all token tensors must share the same embedding dimension")

        if fused is None or totals is None or feature_dim is None:
            continue

        weights = assign_to_cells(coords, centres, tau=tau, backend="python")
        for batch_index in range(batch):
            for token_index, weight_vector in enumerate(weights[batch_index]):
                embedding = tokens[batch_index][token_index]
                for cell_index, weight in enumerate(weight_vector):
                    for dim in range(feature_dim):
                        fused[batch_index][cell_index][dim] += weight * float(embedding[dim])
                    totals[batch_index][cell_index] += weight

    if fused is None or totals is None:
        return [[[] for _ in range(cells)] for _ in range(batch)]

    if agg == "mean":
        for batch_index in range(batch):
            for cell_index in range(cells):
                weight = totals[batch_index][cell_index] or 1.0
                fused[batch_index][cell_index] = [
                    value / weight for value in fused[batch_index][cell_index]
                ]
    return fused


def _materialise_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return [list(item) if isinstance(item, tuple) else item for item in value]
    if value is None:
        return []
    return [value]


__all__ = [
    "aggregate_fields",
    "assign_to_cells",
    "DEFAULT_BACKEND",
    "NUMPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "rasterize_cells",
]
