"""Synthetic doc-understanding benchmark for accuracy vs. token budget."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency for aggregation helpers
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for aggregation helpers
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore[assignment]

from nd_llm.bottleneck import CompressionResult, IBottleneck
from nd_llm.orchestration import Orchestrator, UsageEvent
from nd_llm.stm import STM
from nd_llm.utils import (
    DEFAULT_BACKEND,
    aggregate_fields,
    rasterize_cells,
    OrchestratorConfig,
    STMConfig,
)

from .funsd import (
    build_funsd_encoders,
    build_funsd_registry,
    funsd_fields,
    funsd_numeric_answer_label,
    load_funsd_dataset,
)
from .synthetic import (
    build_invoice_encoders,
    build_invoice_registry,
    high_value_label,
    invoice_fields,
    synthetic_invoice_dataset,
)

FieldsDict = Dict[str, List[MutableMapping[str, Any]]]
FieldsFn = Callable[[Mapping[str, Any]], FieldsDict]
LabelFn = Callable[[Mapping[str, Any]], Any]
PredictFn = Callable[[CompressionResult, Mapping[str, Any]], Any]
MetadataFn = Callable[[Mapping[str, Any], CompressionResult], Mapping[str, Any]]


@dataclass
class BudgetRun:
    """Container summarising a single budget evaluation."""

    budget: int
    accuracy: float
    kept_tokens: List[int]
    budget_probe: Dict[float, Dict[str, Any]]
    retention_probe: Dict[str, Any]
    metrics: Dict[str, float]
    cell_fusions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "budget": self.budget,
            "accuracy": self.accuracy,
            "average_kept_tokens": sum(self.kept_tokens) / len(self.kept_tokens)
            if self.kept_tokens
            else 0.0,
            "budget_probe": self.budget_probe,
            "retention_probe": self.retention_probe,
            "metrics": dict(self.metrics),
        }
        if self.cell_fusions:
            payload["cell_fusions"] = [_serialise_cell_fusion(item) for item in self.cell_fusions]
        return payload


def run_benchmark(
    budget_values: Iterable[int] = (2, 4, 6, 8),
    *,
    dataset_size: int = 24,
    threshold: float = 500.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate token budget trade-offs on a synthetic invoice dataset."""

    registry = build_invoice_registry()
    encoders = build_invoice_encoders(registry)
    dataset = synthetic_invoice_dataset(dataset_size, seed=seed)

    runs: List[BudgetRun] = []
    predict_fn = _invoice_prediction_factory(threshold)
    for budget in budget_values:
        budget_run = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            fields_fn=invoice_fields,
            label_fn=lambda invoice, thr=threshold: high_value_label(invoice, thr),
            predict_fn=predict_fn,
            metadata_fn=_invoice_metadata,
            policy_name="synthetic-doc-benchmark",
            retention_probe_sample_size=5,
            seed=seed,
        )
        runs.append(budget_run)

    return {
        "dataset_size": dataset_size,
        "threshold": threshold,
        "budgets": [run.to_dict() for run in runs],
    }


def run_funsd_benchmark(
    budget_values: Iterable[int] = (8, 12, 16),
    *,
    dataset_size: int = 12,
    split: str = "train",
    data_root: Optional[Path | str] = None,
    use_sample: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate FUNSD documents for numeric-answer retention under budget constraints."""

    registry = build_funsd_registry()
    build_funsd_encoders(registry)
    limit = dataset_size if dataset_size > 0 else None
    dataset = load_funsd_dataset(data_root, split=split, limit=limit, use_sample=use_sample)
    actual_size = len(dataset)

    runs: List[BudgetRun] = []
    for budget in budget_values:
        budget_run = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            fields_fn=funsd_fields,
            label_fn=funsd_numeric_answer_label,
            predict_fn=_funsd_predict_numeric_answer,
            metadata_fn=_funsd_metadata,
            policy_name="funsd-doc-benchmark",
            retention_probe_sample_size=3,
            seed=seed,
        )
        runs.append(budget_run)

    return {
        "dataset": "FUNSD",
        "split": split,
        "dataset_size": actual_size,
        "use_sample": bool(use_sample),
        "budgets": [run.to_dict() for run in runs],
    }


def _evaluate_budget(
    *,
    budget: int,
    dataset: Sequence[Mapping[str, Any]],
    registry_encoders: Mapping[str, Any],
    fields_fn: FieldsFn,
    label_fn: LabelFn,
    predict_fn: PredictFn,
    metadata_fn: Optional[MetadataFn],
    policy_name: str,
    retention_probe_sample_size: int,
    seed: int,
) -> BudgetRun:
    bottleneck = IBottleneck(target_budget=int(budget))

    with TemporaryDirectory(prefix="ndllm-bench-") as tmp:
        storage_dir = Path(tmp)
        stm = STM(STMConfig(storage_dir=storage_dir))
        orchestrator = Orchestrator(
            stm=stm,
            config=OrchestratorConfig(
                target_budget=float(budget),
                policy_name=policy_name,
                budget_step=1.0,
                retention_probe_sample_size=int(retention_probe_sample_size),
            ),
        )

        correct = 0
        kept_tokens: List[int] = []
        metric_totals: Dict[str, float] = defaultdict(float)
        cell_fusions: List[Dict[str, Any]] = []
        for doc_index, document in enumerate(dataset):
            fields = fields_fn(document)
            cell_fusion = _build_cell_fusion(
                document=document,
                fields=fields,
                encoders=registry_encoders,
                doc_index=doc_index,
            )
            cell_fusions.append(cell_fusion)
            result = bottleneck.compress(fields, encoders=registry_encoders)
            prediction = predict_fn(result, document)
            label = label_fn(document)
            if prediction == label:
                correct += 1

            kept = sum(len(indices) for indices in result.telemetry.selected_indices.values())
            kept_tokens.append(kept)

            scores_vector = _scores_to_tensor(result.telemetry.selected_scores)
            metadata: Dict[str, Any] = {
                "budget": budget,
                "kept_tokens": kept,
                "label": label,
                "prediction": prediction,
            }
            if metadata_fn is not None:
                extra = metadata_fn(document, result)
                if extra:
                    metadata.update(dict(extra))
            orchestrator.log_usage_event(
                UsageEvent(
                    tensor=[scores_vector],
                    metadata=metadata,
                )
            )

            for key, value in result.metrics.items():
                metric_totals[key] += float(value)

        accuracy = correct / len(dataset) if dataset else 0.0
        metrics = {key: value / len(dataset) for key, value in metric_totals.items()} if dataset else {}
        budget_probe = orchestrator.budget_sweep()
        retention_probe = orchestrator.run_retention_probe()

    return BudgetRun(
        budget=budget,
        accuracy=accuracy,
        kept_tokens=kept_tokens,
        budget_probe=budget_probe,
        retention_probe=retention_probe,
        metrics=metrics,
        cell_fusions=cell_fusions,
    )


def _invoice_prediction_factory(threshold: float) -> PredictFn:
    def _predict(result: CompressionResult, _: Mapping[str, Any]) -> bool:
        amount_field = result.compressed_fields.get("amount", [])
        return _predict_high_value(amount_field, threshold)

    return _predict


def _invoice_metadata(invoice: Mapping[str, Any], _: CompressionResult) -> Mapping[str, Any]:
    return {"doc_id": invoice.get("doc_id")}


def _predict_high_value(amount_field: Sequence[Any], threshold: float) -> bool:
    for item in amount_field:
        if isinstance(item, Mapping) and "amount" in item:
            if float(item["amount"]) >= threshold:
                return True
        else:
            try:
                if float(item) >= threshold:
                    return True
            except Exception:
                continue
    return False


def _scores_to_tensor(scores: Mapping[str, Sequence[float]]) -> List[float]:
    vector: List[float] = []
    for field in sorted(scores):
        vector.extend(float(value) for value in scores[field])
    if not vector:
        vector.append(0.0)
    return vector


def _build_cell_fusion(
    *,
    document: Mapping[str, Any],
    fields: Mapping[str, Sequence[MutableMapping[str, Any]]],
    encoders: Mapping[str, Any],
    doc_index: int,
    grid_hw: tuple[int, int] = (24, 16),
    backend: Optional[str] = None,
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
    base_centers, coord_maps = _layout_coordinate_maps(layout_entries)

    field_entries: List[Dict[str, Any]] = []
    for field, embeddings in encoded.items():
        if not embeddings:
            continue
        coords = _resolve_field_coordinates(
            field=field,
            entries=list(fields.get(field, [])),
            coord_maps=coord_maps,
            base_centers=base_centers,
            count=len(embeddings),
        )
        padded = _pad_field_embeddings(embeddings, target_dim)
        tokens_array = _to_backend_array(padded, backend)
        coords_array = _to_backend_array(coords, backend)
        field_entries.append({"tokens": tokens_array, "coords": coords_array})

    fused = aggregate_fields(field_entries, cell_centers, agg="mean", backend=backend)

    document_id = (
        document.get("doc_id")
        or document.get("id")
        or document.get("document_id")
        or doc_index
    )

    return {
        "document_id": str(document_id),
        "document_index": int(doc_index),
        "backend": backend,
        "grid_hw": list(grid_hw),
        "feature_dim": target_dim,
        "cells": _to_serialisable(fused),
        "centers": _to_serialisable(cell_centers),
    }


def _layout_coordinate_maps(
    layout_entries: Sequence[MutableMapping[str, Any]] | None,
) -> tuple[List[List[float]], Dict[str, Dict[str, List[float]]]]:
    base_centers: List[List[float]] = []
    token_map: Dict[str, List[float]] = {}
    line_map: Dict[str, List[float]] = {}
    entries = list(layout_entries or [])
    for index, entry in enumerate(entries):
        centre = _center_from_entry(entry)
        base_centers.append(centre)
        key = _identifier_key(index)
        token_map.setdefault(key, centre)
        line_map.setdefault(key, centre)
        if isinstance(entry, Mapping):
            token_id = entry.get("token_id")
            if token_id is not None:
                token_map[_identifier_key(token_id)] = centre
            line_id = entry.get("line_id")
            if line_id is not None:
                line_map[_identifier_key(line_id)] = centre
    return base_centers, {"token": token_map, "line": line_map}


def _resolve_field_coordinates(
    *,
    field: str,
    entries: Sequence[MutableMapping[str, Any]] | None,
    coord_maps: Mapping[str, Mapping[str, Sequence[float]]],
    base_centers: Sequence[Sequence[float]],
    count: int,
) -> List[List[float]]:
    coords: List[List[float]] = []
    fallback = list(base_centers) or [[0.5, 0.5]]
    items = list(entries or [])
    token_map = dict(coord_maps.get("token", {}))
    line_map = dict(coord_maps.get("line", {}))

    for index in range(count):
        centre: Sequence[float] | None = None
        entry = items[index] if index < len(items) else None
        if isinstance(entry, Mapping):
            token_ids = entry.get("token_ids")
            if isinstance(token_ids, Sequence) and token_ids:
                collected = [
                    token_map.get(_identifier_key(token_id))
                    for token_id in token_ids
                    if token_map.get(_identifier_key(token_id)) is not None
                ]
                if collected:
                    centre = _mean_coords(collected)
            if centre is None:
                token_id = entry.get("token_id")
                if token_id is not None:
                    centre = token_map.get(_identifier_key(token_id))
            if centre is None:
                line_id = entry.get("line_id")
                if line_id is not None:
                    centre = line_map.get(_identifier_key(line_id))
        if centre is None and index < len(base_centers):
            centre = base_centers[index]
        if centre is None:
            centre = fallback[min(index, len(fallback) - 1)]
        coords.append([float(centre[0]), float(centre[1])])
    return coords


def _pad_field_embeddings(vectors: Sequence[Sequence[float]], target_dim: int) -> List[List[float]]:
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
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(item) for item in value]
    return value


def _center_from_entry(entry: Mapping[str, Any] | None) -> List[float]:
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


def _identifier_key(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except Exception:
            return str(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    return str(value)


def _mean_coords(values: Sequence[Sequence[float]]) -> List[float]:
    if not values:
        return [0.5, 0.5]
    total_y = sum(float(coord[0]) for coord in values)
    total_x = sum(float(coord[1]) for coord in values)
    length = float(len(values)) or 1.0
    return [total_y / length, total_x / length]


def _serialise_cell_fusion(fusion: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": str(fusion.get("document_id", "")),
        "document_index": int(fusion.get("document_index", 0)),
        "backend": fusion.get("backend"),
        "grid_hw": list(fusion.get("grid_hw", [])),
        "feature_dim": int(fusion.get("feature_dim", 0)),
        "cells": _copy_nested_list(fusion.get("cells", [])),
        "centers": _copy_nested_list(fusion.get("centers", [])),
    }


def _copy_nested_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_copy_nested_list(item) for item in value]
    return value


def _funsd_predict_numeric_answer(result: CompressionResult, _: Mapping[str, Any]) -> bool:
    for entity in result.compressed_fields.get("entity", []):
        if isinstance(entity, Mapping) and str(entity.get("label", "")).lower() == "answer":
            text = entity.get("text", "")
            if any(char.isdigit() for char in str(text)):
                return True
    for token in result.compressed_fields.get("text", []):
        if not isinstance(token, Mapping):
            continue
        if token.get("is_answer") and any(char.isdigit() for char in str(token.get("text", ""))):
            return True
    return False


def _funsd_metadata(document: Mapping[str, Any], _: CompressionResult) -> Mapping[str, Any]:
    doc_id = document.get("doc_id") or document.get("id")
    entities = document.get("form", [])
    answer_count = sum(1 for item in entities if str(item.get("label", "")).lower() == "answer")
    return {
        "doc_id": doc_id,
        "entity_count": len(entities),
        "answer_entities": answer_count,
    }


def main() -> None:
    """CLI entry point printing a JSON benchmark report."""

    report = run_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["run_benchmark", "run_funsd_benchmark"]
