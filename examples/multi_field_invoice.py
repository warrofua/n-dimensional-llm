"""End-to-end multi-field invoice demo using the ND-LLM stubs."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nd_llm.bottleneck import IBottleneck
from nd_llm.orchestration import CompressionRecord, Orchestrator, UsageEvent
from nd_llm.stm import STM
from nd_llm.utils import (
    DEFAULT_BACKEND,
    aggregate_fields,
    build_mi_proxy_context,
    rasterize_cells,
    OrchestratorConfig,
    STMConfig,
)

from benchmarks.synthetic import (
    build_invoice_encoders,
    build_invoice_registry,
    invoice_fields,
    synthetic_invoice,
)

try:  # pragma: no cover - optional dependency in tests
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - fallback when numpy is missing
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency in tests
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None  # type: ignore[assignment]


def run_demo(
    storage_dir: Path | None = None,
    *,
    budget: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the multi-field invoice demo and return a summary payload."""

    registry = build_invoice_registry()
    encoders = build_invoice_encoders(registry)
    invoice = synthetic_invoice(doc_id=1, num_lines=6, seed=seed)
    fields = invoice_fields(invoice)
    cell_fusion = _fuse_invoice_cells(fields, encoders)

    bottleneck = IBottleneck(target_budget=budget)
    mi_proxy, mi_context = build_mi_proxy_context(
        fields,
        registry.encoders,
        preferred_fields=("text", "layout", "amount"),
    )
    compression = bottleneck.compress(
        fields,
        encoders=registry.encoders,
        context=mi_context,
        mi_proxy=mi_proxy,
    )
    reconstructed = bottleneck.decompress(compression)

    def _execute(storage_path: Path) -> Dict[str, Any]:
        stm = STM(STMConfig(storage_dir=storage_path))
        orchestrator = Orchestrator(
            stm=stm,
            config=OrchestratorConfig(
                target_budget=float(budget),
                policy_name="example-invoice",
                budget_step=1.0,
                retention_probe_sample_size=5,
            ),
        )

        scores_vector = _scores_to_tensor(compression.telemetry.selected_scores)
        kept_tokens = sum(len(v) for v in compression.telemetry.selected_indices.values())
        stm_key = orchestrator.log_usage_event(
            UsageEvent(
                tensor=[scores_vector],
                metadata={
                    "doc_id": invoice["doc_id"],
                    "vendor": invoice["vendor"],
                    "kept_tokens": kept_tokens,
                },
                compression=CompressionRecord.from_result(
                    compression,
                    bottleneck=bottleneck,
                ),
            )
        )

        stored_tensor, stored_metadata = stm.retrieve(stm_key)
        if hasattr(stored_tensor, "tolist"):
            stored_tensor = stored_tensor.tolist()
        budget_probe = orchestrator.budget_sweep()
        retention_probe = orchestrator.run_retention_probe()

        return {
            "registry": registry,
            "invoice": invoice,
            "fields": fields,
            "compression": compression,
            "reconstructed": reconstructed,
            "stm_key": stm_key,
            "stm_entry": {"tensor": stored_tensor, "metadata": stored_metadata},
            "budget_probe": budget_probe,
            "retention_probe": retention_probe,
            "usage_log": orchestrator.usage_log,
            "cell_fusion": cell_fusion,
        }

    if storage_dir is None:
        with TemporaryDirectory(prefix="ndllm-example-") as tmp:
            return _execute(Path(tmp))
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    return _execute(storage_path)


def _scores_to_tensor(scores: Mapping[str, Sequence[float]]) -> Sequence[float]:
    vector: list[float] = []
    for field in sorted(scores):
        vector.extend(float(value) for value in scores[field])
    if not vector:
        vector.append(0.0)
    return vector


def _fuse_invoice_cells(
    fields: Mapping[str, Sequence[MutableMapping[str, Any]]],
    encoders: Mapping[str, Any],
    *,
    grid_hw: tuple[int, int] = (16, 12),
    backend: str | None = None,
) -> Dict[str, Any]:
    backend = backend or DEFAULT_BACKEND
    encoded: Dict[str, List[List[float]]] = {}
    feature_dims: List[int] = []
    for field, encoder in encoders.items():
        batch = list(fields.get(field, []))
        if not batch:
            encoded[field] = []
            continue
        embeddings = encoder.encode(batch)
        vectors = [list(vector) for vector in embeddings]
        encoded[field] = vectors
        feature_dims.extend(len(vector) for vector in vectors if vector)

    target_dim = max(feature_dims, default=0)
    cell_centers = rasterize_cells(1, grid_hw=grid_hw, backend=backend)

    layout_entries = list(fields.get("layout", []))
    base_centers, line_map = _invoice_layout_centers(layout_entries)

    field_entries: List[Dict[str, Any]] = []
    for field, embeddings in encoded.items():
        if not embeddings:
            continue
        coords = _invoice_field_coords(
            field,
            list(fields.get(field, [])),
            base_centers,
            line_map,
            len(embeddings),
        )
        padded = _pad_embeddings(embeddings, target_dim)
        tokens_array = _to_backend_array(padded, backend)
        coords_array = _to_backend_array(coords, backend)
        field_entries.append({"tokens": tokens_array, "coords": coords_array})

    fused = aggregate_fields(field_entries, cell_centers, agg="mean", backend=backend)

    return {
        "cells": _to_serialisable(fused),
        "centers": _to_serialisable(cell_centers),
        "backend": backend,
        "grid_hw": list(grid_hw),
        "feature_dim": target_dim,
    }


def _invoice_layout_centers(
    layout_entries: Sequence[MutableMapping[str, Any]] | None,
) -> tuple[List[List[float]], Dict[str, List[float]]]:
    centres: List[List[float]] = []
    line_map: Dict[str, List[float]] = {}
    entries = list(layout_entries or [])
    for index, entry in enumerate(entries):
        centre = _entry_center(entry)
        centres.append(centre)
        line_id = entry.get("line_id") if isinstance(entry, Mapping) else None
        if line_id is not None:
            line_map[str(int(line_id))] = centre
        line_map.setdefault(str(index), centre)
    return centres, line_map


def _invoice_field_coords(
    field: str,
    entries: Sequence[MutableMapping[str, Any]] | None,
    base_centers: Sequence[Sequence[float]],
    line_map: Mapping[str, Sequence[float]],
    count: int,
) -> List[List[float]]:
    coords: List[List[float]] = []
    fallback = list(base_centers) or [[0.5, 0.5]]
    items = list(entries or [])
    for index in range(count):
        centre: Sequence[float] | None = None
        entry = items[index] if index < len(items) else None
        if isinstance(entry, Mapping):
            line_id = entry.get("line_id")
            if line_id is not None and str(int(line_id)) in line_map:
                centre = line_map[str(int(line_id))]
        if centre is None and index < len(base_centers):
            centre = base_centers[index]
        if centre is None:
            centre = fallback[min(index, len(fallback) - 1)]
        coords.append([float(centre[0]), float(centre[1])])
    return coords


def _entry_center(entry: Mapping[str, Any] | None) -> List[float]:
    if isinstance(entry, Mapping):
        xyxy = entry.get("xyxy") or entry.get("bbox") or entry.get("box")
        if isinstance(xyxy, Mapping):
            xyxy = [
                xyxy.get("x1", 0.0),
                xyxy.get("y1", 0.0),
                xyxy.get("x2", 1.0),
                xyxy.get("y2", 1.0),
            ]
        if isinstance(xyxy, Sequence) and len(xyxy) >= 4:
            x1, y1, x2, y2 = (float(xyxy[i]) for i in range(4))
            centre_y = (y1 + y2) / 2.0
            centre_x = (x1 + x2) / 2.0
            return [centre_y, centre_x]
    return [0.5, 0.5]


def _pad_embeddings(vectors: Sequence[Sequence[float]], target_dim: int) -> List[List[float]]:
    if target_dim <= 0:
        return [[0.0] * 0 for _ in vectors]
    padded: List[List[float]] = []
    for vector in vectors:
        trimmed = [float(value) for value in vector[:target_dim]]
        if len(trimmed) < target_dim:
            trimmed.extend([0.0] * (target_dim - len(trimmed)))
        padded.append(trimmed)
    return padded


def _to_backend_array(data: Sequence[Sequence[float]], backend: str) -> Any:
    if backend == "torch":
        if torch is None:  # pragma: no cover - defensive
            raise RuntimeError("torch backend requested but torch is unavailable")
        tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor
    if backend == "numpy":
        if _np is None:  # pragma: no cover - defensive
            raise RuntimeError("numpy backend requested but numpy is unavailable")
        array = _np.asarray(data, dtype=_np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        return array
    nested = [[float(value) for value in row] for row in data]
    return [nested]


def _to_serialisable(value: Any) -> Any:
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if _np is not None and isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return list(value)


def main() -> None:
    """CLI entry point that prints demo artefacts."""

    summary = run_demo()
    print("Compression metrics:")
    pprint(summary["compression"].metrics)
    print("\nSTM metadata:")
    pprint(summary["stm_entry"]["metadata"])
    print("\nBudget probe:")
    pprint(summary["budget_probe"])
    print("\nRetention probe:")
    pprint(summary["retention_probe"])


if __name__ == "__main__":
    main()
