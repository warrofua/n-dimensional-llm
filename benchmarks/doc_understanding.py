"""Synthetic doc-understanding benchmark for accuracy vs. token budget."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    build_mi_proxy_context,
    rasterize_cells,
    OrchestratorConfig,
    STMConfig,
)

from .doclaynet import (
    build_doclaynet_encoders,
    build_doclaynet_registry,
    doclaynet_contains_table,
    doclaynet_fields,
    load_doclaynet_dataset,
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
AblationFn = Callable[
    [Mapping[str, Any], Mapping[str, Sequence[MutableMapping[str, Any]]], int, int],
    Mapping[str, List[MutableMapping[str, Any]]],
]


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
    ablations: Dict[str, Any] = field(default_factory=dict)

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
        if self.ablations:
            payload["ablations"] = self.ablations
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
    ablations = _invoice_ablation_suite(seed)

    runs: List[BudgetRun] = []
    predict_fn = _invoice_prediction_factory(threshold)

    def _label_fn(invoice: Mapping[str, Any], threshold_value: float = threshold) -> Any:
        return high_value_label(invoice, threshold_value)

    for budget in budget_values:
        budget_run = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            fields_fn=invoice_fields,
            label_fn=_label_fn,
            predict_fn=predict_fn,
            metadata_fn=_invoice_metadata,
            policy_name="synthetic-doc-benchmark",
            retention_probe_sample_size=5,
            seed=seed,
            ablations=ablations,
            mi_field_priorities=("text", "layout", "amount"),
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
    ablations = _funsd_ablation_suite(seed)
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
            ablations=ablations,
            mi_field_priorities=("text", "layout"),
        )
        runs.append(budget_run)

    return {
        "dataset": "FUNSD",
        "split": split,
        "dataset_size": actual_size,
        "use_sample": bool(use_sample),
        "budgets": [run.to_dict() for run in runs],
    }


def run_doclaynet_benchmark(
    budget_values: Iterable[int] = (6, 12),
    *,
    dataset_size: int = 6,
    split: str = "train",
    data_root: Optional[Path | str] = None,
    use_sample: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate DocLayNet documents for table retention under budget constraints."""

    registry = build_doclaynet_registry()
    build_doclaynet_encoders(registry)
    limit = dataset_size if dataset_size > 0 else None
    dataset = load_doclaynet_dataset(data_root, split=split, limit=limit, use_sample=use_sample)
    actual_size = len(dataset)

    runs: List[BudgetRun] = []
    ablations = _doclaynet_ablation_suite(seed)
    for budget in budget_values:
        budget_run = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            fields_fn=doclaynet_fields,
            label_fn=doclaynet_contains_table,
            predict_fn=_doclaynet_predict_contains_table,
            metadata_fn=_doclaynet_metadata,
            policy_name="doclaynet-doc-benchmark",
            retention_probe_sample_size=3,
            seed=seed,
            ablations=ablations,
            mi_field_priorities=("layout", "text", "segment"),
        )
        runs.append(budget_run)

    return {
        "dataset": "DocLayNet",
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
    ablations: Optional[Mapping[str, "AblationFn"]] = None,
    mi_field_priorities: Optional[Sequence[str]] = None,
) -> BudgetRun:
    bottleneck = IBottleneck(target_budget=int(budget))

    ablation_totals: Dict[str, Dict[str, Any]] = {}
    if ablations:
        ablation_totals = {name: _make_ablation_totals() for name in ablations}

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
        total_latency = 0.0
        total_flops = 0.0
        total_pre_distortion = 0.0
        total_post_distortion = 0.0
        label_counts: Counter[Any] = Counter()

        for doc_index, document in enumerate(dataset):
            fields = fields_fn(document)
            cell_fusion = _build_cell_fusion(
                document=document,
                fields=fields,
                encoders=registry_encoders,
                doc_index=doc_index,
            )
            cell_fusions.append(cell_fusion)
            registration_metrics = cell_fusion.get("registration", {})
            total_pre_distortion += float(registration_metrics.get("pre_distortion", 0.0))
            total_post_distortion += float(registration_metrics.get("post_distortion", 0.0))

            label = label_fn(document)
            label_counts[label] += 1

            start_time = time.perf_counter()
            mi_proxy, mi_context = build_mi_proxy_context(
                fields,
                registry_encoders,
                preferred_fields=mi_field_priorities,
            )
            result = bottleneck.compress(
                fields,
                encoders=registry_encoders,
                context=mi_context,
                mi_proxy=mi_proxy,
            )
            latency = time.perf_counter() - start_time
            total_latency += latency
            total_flops += _estimate_encoder_flops(fields, registry_encoders)

            prediction = predict_fn(result, document)
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
                "encoder_latency_seconds": latency,
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

            if ablation_totals:
                for name, ablation_fn in (ablations or {}).items():
                    mutated_fields = ablation_fn(
                        document,
                        _clone_fields(fields),
                        doc_index,
                        seed,
                    )
                    ab_cell_fusion = _build_cell_fusion(
                        document=document,
                        fields=mutated_fields,
                        encoders=registry_encoders,
                        doc_index=doc_index,
                    )
                    ab_start = time.perf_counter()
                    ab_proxy, ab_context = build_mi_proxy_context(
                        mutated_fields,
                        registry_encoders,
                        preferred_fields=mi_field_priorities,
                    )
                    ab_result = bottleneck.compress(
                        mutated_fields,
                        encoders=registry_encoders,
                        context=ab_context,
                        mi_proxy=ab_proxy,
                    )
                    ab_latency = time.perf_counter() - ab_start
                    ab_prediction = predict_fn(ab_result, document)
                    ab_kept = sum(
                        len(indices) for indices in ab_result.telemetry.selected_indices.values()
                    )
                    ab_metrics = ablation_totals[name]
                    ab_metrics["count"] += 1
                    if ab_prediction == label:
                        ab_metrics["correct"] += 1
                    ab_metrics["kept_tokens"].append(ab_kept)
                    ab_metrics["latency"] += ab_latency
                    ab_metrics["flops"] += _estimate_encoder_flops(mutated_fields, registry_encoders)
                    for key, value in ab_result.metrics.items():
                        ab_metrics["metrics"][key] += float(value)
                    ab_reg = ab_cell_fusion.get("registration", {})
                    ab_metrics["reg_pre"] += float(ab_reg.get("pre_distortion", 0.0))
                    ab_metrics["reg_post"] += float(ab_reg.get("post_distortion", 0.0))
                    ab_metrics["label_counts"][label] += 1

        dataset_size = len(dataset)
        accuracy = correct / dataset_size if dataset_size else 0.0
        metrics = {key: value / dataset_size for key, value in metric_totals.items()} if dataset_size else {}

        if dataset_size:
            metrics["encoder_latency_seconds"] = total_latency / dataset_size
            metrics["encoder_flops"] = total_flops / dataset_size
            metrics["registration_pre_distortion"] = total_pre_distortion / dataset_size
            metrics["registration_post_distortion"] = total_post_distortion / dataset_size

        label_entropy = _entropy(label_counts)
        metrics["label_entropy"] = label_entropy
        mi_estimate = metrics.get("mi_lower_bound") or metrics.get("ib_proxy") or 0.0
        conditional_entropy = max(0.0, label_entropy - mi_estimate)
        metrics["conditional_entropy"] = conditional_entropy
        metrics["fano_error_bound"] = _fano_lower_bound(conditional_entropy, len(label_counts))

        budget_probe = orchestrator.budget_sweep()
        retention_probe = orchestrator.run_retention_probe()

    ablation_report = _summarise_ablations(ablation_totals)

    return BudgetRun(
        budget=budget,
        accuracy=accuracy,
        kept_tokens=kept_tokens,
        budget_probe=budget_probe,
        retention_probe=retention_probe,
        metrics=metrics,
        cell_fusions=cell_fusions,
        ablations=ablation_report,
    )


def _make_ablation_totals() -> Dict[str, Any]:
    return {
        "correct": 0,
        "count": 0,
        "kept_tokens": [],
        "metrics": defaultdict(float),
        "latency": 0.0,
        "flops": 0.0,
        "reg_pre": 0.0,
        "reg_post": 0.0,
        "label_counts": Counter(),
    }


def _summarise_ablations(totals: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for name, values in totals.items():
        count = int(values.get("count", 0))
        if count <= 0:
            continue
        metrics_dict = values.get("metrics", {})
        averaged = {key: float(val) / count for key, val in metrics_dict.items()}
        averaged["encoder_latency_seconds"] = float(values.get("latency", 0.0)) / count
        averaged["encoder_flops"] = float(values.get("flops", 0.0)) / count
        averaged["registration_pre_distortion"] = float(values.get("reg_pre", 0.0)) / count
        averaged["registration_post_distortion"] = float(values.get("reg_post", 0.0)) / count

        label_counts = values.get("label_counts", Counter())
        averaged["label_entropy"] = _entropy(label_counts)
        mi_est = averaged.get("mi_lower_bound") or averaged.get("ib_proxy") or 0.0
        conditional = max(0.0, averaged["label_entropy"] - mi_est)
        averaged["conditional_entropy"] = conditional
        averaged["fano_error_bound"] = _fano_lower_bound(conditional, len(label_counts))

        kept_tokens: Sequence[int] = values.get("kept_tokens", [])
        report[name] = {
            "accuracy": float(values.get("correct", 0)) / count,
            "count": count,
            "average_kept_tokens": sum(kept_tokens) / len(kept_tokens)
            if kept_tokens
            else 0.0,
            "metrics": averaged,
        }
    return report


def _estimate_encoder_flops(
    fields: Mapping[str, Sequence[Any]],
    encoders: Mapping[str, Any],
) -> float:
    total = 0.0
    for field, entries in fields.items():
        encoder = encoders.get(field)
        if encoder is None:
            continue
        dim = getattr(encoder, "embedding_dim", 1)
        count = len(entries)
        total += float(count * max(dim, 1) * max(dim, 1))
    return total


def _entropy(counts: Mapping[Any, int]) -> float:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return float(entropy)


def _fano_lower_bound(conditional_entropy: float, num_labels: int) -> float:
    if num_labels <= 1:
        return 0.0
    denom = math.log2(float(num_labels))
    if denom == 0:
        return 0.0
    return max(0.0, (conditional_entropy - 1.0) / denom)


def _clone_fields(
    fields: Mapping[str, Sequence[MutableMapping[str, Any]]]
) -> Dict[str, List[MutableMapping[str, Any]]]:
    materialised = {name: list(entries) for name, entries in fields.items()}
    return deepcopy(materialised)


def _drop_field_ablation(field: str) -> AblationFn:
    def _apply(
        _: Mapping[str, Any],
        fields: Mapping[str, Sequence[MutableMapping[str, Any]]],
        __: int,
        ___: int,
    ) -> Mapping[str, List[MutableMapping[str, Any]]]:
        mutated: Dict[str, List[MutableMapping[str, Any]]] = {
            str(name): list(entries) for name, entries in fields.items()
        }
        mutated[field] = []
        return mutated

    return _apply


def _noise_field_ablation(
    field: str,
    value_key: str,
    *,
    scale: float,
) -> AblationFn:
    def _apply(
        _: Mapping[str, Any],
        fields: Mapping[str, Sequence[MutableMapping[str, Any]]],
        doc_index: int,
        seed: int,
    ) -> Mapping[str, List[MutableMapping[str, Any]]]:
        mutated: Dict[str, List[MutableMapping[str, Any]]] = {
            str(name): [dict(entry) for entry in entries]
            for name, entries in fields.items()
        }
        rng = random.Random(seed * 97 + doc_index * 13 + hash(field))
        updated: List[MutableMapping[str, Any]] = []
        for entry in mutated.get(field, []):
            value = entry.get(value_key)
            if value is None:
                updated.append(entry)
                continue
            try:
                base = float(value)
            except (TypeError, ValueError):
                updated.append(entry)
                continue
            noise = rng.uniform(-scale, scale)
            entry[value_key] = base + noise
            updated.append(entry)
        mutated[field] = updated
        return mutated

    return _apply


def _perturb_layout_ablation(*, scale: float, field: str = "layout") -> AblationFn:
    def _apply(
        _: Mapping[str, Any],
        fields: Mapping[str, Sequence[MutableMapping[str, Any]]],
        doc_index: int,
        seed: int,
    ) -> Mapping[str, List[MutableMapping[str, Any]]]:
        mutated: Dict[str, List[MutableMapping[str, Any]]] = {
            str(name): [dict(entry) for entry in entries]
            for name, entries in fields.items()
        }
        rng = random.Random(seed * 137 + doc_index * 19)
        perturbed: List[MutableMapping[str, Any]] = []
        for entry in mutated.get(field, []):
            mutated_entry = dict(entry)
            xyxy = list(mutated_entry.get("xyxy", []))
            if len(xyxy) >= 4:
                for idx in range(4):
                    jitter = rng.uniform(-scale, scale)
                    value = float(xyxy[idx]) if idx < len(xyxy) else 0.0
                    xyxy[idx] = min(1.0, max(0.0, value + jitter))
                mutated_entry["xyxy"] = xyxy
            perturbed.append(mutated_entry)
        mutated[field] = perturbed
        return mutated

    return _apply


def _shuffle_field_ablation(field: str) -> AblationFn:
    def _apply(
        _: Mapping[str, Any],
        fields: Mapping[str, Sequence[MutableMapping[str, Any]]],
        doc_index: int,
        seed: int,
    ) -> Mapping[str, List[MutableMapping[str, Any]]]:
        mutated: Dict[str, List[MutableMapping[str, Any]]] = {
            str(name): [dict(entry) for entry in entries]
            for name, entries in fields.items()
        }
        rng = random.Random(seed * 59 + doc_index * 17 + hash(field))
        entries = mutated.get(field, [])
        rng.shuffle(entries)
        mutated[field] = entries
        return mutated

    return _apply


def _invoice_ablation_suite(seed: int) -> Dict[str, AblationFn]:
    return {
        "drop_amount": _drop_field_ablation("amount"),
        "drop_layout": _drop_field_ablation("layout"),
        "noise_amount": _noise_field_ablation("amount", "amount", scale=25.0),
        "perturb_layout": _perturb_layout_ablation(scale=0.05),
        "shuffle_text": _shuffle_field_ablation("text"),
    }


def _funsd_ablation_suite(seed: int) -> Dict[str, AblationFn]:
    return {
        "drop_layout": _drop_field_ablation("layout"),
        "drop_text": _drop_field_ablation("text"),
        "perturb_layout": _perturb_layout_ablation(scale=0.02),
        "shuffle_entities": _shuffle_field_ablation("entity"),
    }


def _doclaynet_ablation_suite(seed: int) -> Dict[str, AblationFn]:
    return {
        "drop_layout": _drop_field_ablation("layout"),
        "drop_text": _drop_field_ablation("text"),
        "perturb_layout": _perturb_layout_ablation(scale=0.03),
        "shuffle_segments": _shuffle_field_ablation("segment"),
    }


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
    all_coords: List[List[float]] = []
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
        all_coords.extend(coords)
        padded = _pad_field_embeddings(embeddings, target_dim)
        tokens_array = _to_backend_array(padded, backend)
        coords_array = _to_backend_array(coords, backend)
        field_entries.append({"tokens": tokens_array, "coords": coords_array})

    fused = aggregate_fields(field_entries, cell_centers, agg="mean", backend=backend)

    registration = _registration_metrics(layout_entries, all_coords, cell_centers)

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
        "registration": registration,
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
                collected: List[Sequence[float]] = []
                for token_id in token_ids:
                    resolved = token_map.get(_identifier_key(token_id))
                    if resolved is not None:
                        collected.append(resolved)
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


def _registration_metrics(
    layout_entries: Sequence[MutableMapping[str, Any]],
    field_coords: Sequence[Sequence[float]],
    cell_centers: Any,
) -> Dict[str, Any]:
    layout_centers = [_center_from_entry(entry) for entry in layout_entries if entry is not None]
    canonical_centers = _flatten_centers(cell_centers)
    pre = _average_nearest_distance(layout_centers, canonical_centers)
    post = _average_nearest_distance(field_coords, canonical_centers)
    return {
        "pre_distortion": pre,
        "post_distortion": post if field_coords else pre,
        "layout_tokens": len(layout_centers),
        "field_tokens": len(field_coords),
    }


def _flatten_centers(centers: Any) -> List[List[float]]:
    serialised = _to_serialisable(centers)
    flattened: List[List[float]] = []
    for batch in serialised:
        for coord in batch:
            if isinstance(coord, Sequence) and len(coord) >= 2:
                flattened.append([float(coord[0]), float(coord[1])])
    return flattened


def _average_nearest_distance(
    points: Sequence[Sequence[float]],
    centres: Sequence[Sequence[float]],
) -> float:
    if not points or not centres:
        return 0.0
    total = 0.0
    for point in points:
        distances = [_euclidean_distance(point, centre) for centre in centres]
        total += min(distances) if distances else 0.0
    return total / max(len(points), 1)


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    dy = float(a[0]) - float(b[0])
    dx = float(a[1]) - float(b[1])
    return math.sqrt(dy * dy + dx * dx)


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
    payload = {
        "document_id": str(fusion.get("document_id", "")),
        "document_index": int(fusion.get("document_index", 0)),
        "backend": fusion.get("backend"),
        "grid_hw": list(fusion.get("grid_hw", [])),
        "feature_dim": int(fusion.get("feature_dim", 0)),
        "cells": _copy_nested_list(fusion.get("cells", [])),
        "centers": _copy_nested_list(fusion.get("centers", [])),
    }
    registration = fusion.get("registration")
    if isinstance(registration, Mapping):
        payload["registration"] = {
            "pre_distortion": float(registration.get("pre_distortion", 0.0)),
            "post_distortion": float(registration.get("post_distortion", 0.0)),
            "layout_tokens": int(registration.get("layout_tokens", 0)),
            "field_tokens": int(registration.get("field_tokens", 0)),
        }
    return payload


def _copy_nested_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_copy_nested_list(item) for item in value]
    return value


def _doclaynet_predict_contains_table(result: CompressionResult, _: Mapping[str, Any]) -> bool:
    segments = result.compressed_fields.get("segment")
    if segments is None:
        segments = result.compressed_fields.get("region", [])
    for segment in segments:
        if isinstance(segment, Mapping):
            label = (
                segment.get("label")
                or segment.get("segment_label")
                or segment.get("region_label")
            )
        else:
            label = segment
        if isinstance(label, str) and label.lower() == "table":
            return True
    for token in result.compressed_fields.get("text", []):
        value: Any
        if isinstance(token, Mapping):
            value = token.get("text")
        else:
            value = token
        if isinstance(value, str) and "table" in value.lower():
            return True
    return False


def _doclaynet_metadata(document: Mapping[str, Any], result: CompressionResult) -> Mapping[str, Any]:
    doc_id = document.get("doc_id") or document.get("id")
    segments = result.compressed_fields.get("segment")
    if segments is None:
        segments = result.compressed_fields.get("region", [])
    kept_segments = []
    for segment in segments:
        if isinstance(segment, Mapping):
            label = (
                segment.get("label")
                or segment.get("segment_label")
                or segment.get("region_label")
            )
            segment_id = (
                segment.get("segment_id")
                if segment.get("segment_id") is not None
                else segment.get("region_id")
            )
        else:
            label = segment
            segment_id = None
        kept_segments.append(
            {
                "segment_id": segment_id,
                "region_id": segment_id,
                "label": str(label) if label is not None else "",
            }
        )

    metadata = {
        "doc_id": doc_id,
        "contains_table": doclaynet_contains_table(document),
        "kept_segments": kept_segments,
        "kept_segment_count": len(kept_segments),
    }
    # Preserve legacy keys for downstream consumers expecting the old naming.
    metadata["kept_regions"] = [
        {"region_id": entry.get("region_id"), "label": entry.get("label", "")}
        for entry in kept_segments
    ]
    metadata["kept_region_count"] = metadata["kept_segment_count"]
    return metadata


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


__all__ = ["run_benchmark", "run_funsd_benchmark", "run_doclaynet_benchmark"]
