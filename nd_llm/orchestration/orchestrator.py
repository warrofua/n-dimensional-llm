"""Orchestration utilities coordinating STM persistence and policy sweeps."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from nd_llm.bottleneck.ib import CompressionResult, CompressionTelemetry, IBottleneck
from nd_llm.constraints import ConstraintModule, ConstraintResult
from nd_llm.orchestration.budget import (
    BudgetDecision,
    BudgetObservation,
    BudgetStrategy,
    CompressionRatioBudgetStrategy,
)
from nd_llm.stm import STM, TensorLike
from nd_llm.utils.config import OrchestratorConfig

def _to_serialisable(value: Any) -> Any:
    """Convert values to JSON-serialisable structures."""

    if hasattr(value, "tolist"):
        try:
            return _to_serialisable(value.tolist())
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {str(k): _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return value


def _iterable_to_list(value: Any) -> List[Any]:
    """Normalise common iterable types (including numpy arrays) to a list."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return [item for item in value]
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _coerce_int(value: Any) -> int:
    """Best-effort conversion of arbitrary values to integers."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    """Best-effort conversion of arbitrary values to floats."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compute_layout_signature(idx_cells: Mapping[str, Any]) -> Optional[str]:
    canonical = idx_cells.get("canonical") if isinstance(idx_cells, Mapping) else None
    if not isinstance(canonical, Mapping):
        return None
    parts: List[str] = []
    for field in sorted(canonical):
        values = _iterable_to_list(canonical[field])
        if not values:
            continue
        part = f"{field}:{','.join(str(value) for value in values)}"
        parts.append(part)
    if not parts:
        return None
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class BudgetCandidate:
    """Candidate policy describing a field set and associated budget."""

    fields: Tuple[str, ...]
    budget: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fields = tuple(str(field) for field in self.fields)
        self.budget = float(self.budget)
        self.metadata = {str(k): v for k, v in dict(self.metadata).items()}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fields": list(self.fields),
            "budget": float(self.budget),
            "metadata": _to_serialisable(self.metadata),
        }


class BudgetMetaModel:
    """Interface for scoring candidate budgets using telemetry and MI."""

    name: str = "meta-model"

    def score_candidate(
        self,
        candidate: BudgetCandidate,
        *,
        history: Sequence["CompressionRecord"],
        observations: Sequence[BudgetObservation],
        features: Mapping[str, Any],
    ) -> float:
        raise NotImplementedError


class HeuristicBudgetMetaModel(BudgetMetaModel):
    """Heuristic scorer combining mutual information and utilisation signals."""

    name = "heuristic-budget-v0"

    def __init__(
        self,
        *,
        information_weight: float = 0.6,
        field_weight: float = 0.3,
        utilisation_weight: float = 0.5,
        pressure_weight: float = 1.2,
        diversity_weight: float = 0.05,
        history_weight: float = 0.1,
        min_balance: float = 0.2,
    ) -> None:
        self.information_weight = float(information_weight)
        self.field_weight = float(field_weight)
        self.utilisation_weight = float(utilisation_weight)
        self.pressure_weight = float(pressure_weight)
        self.diversity_weight = float(diversity_weight)
        self.history_weight = float(history_weight)
        self.min_balance = float(min_balance)

    def score_candidate(
        self,
        candidate: BudgetCandidate,
        *,
        history: Sequence["CompressionRecord"],
        observations: Sequence[BudgetObservation],
        features: Mapping[str, Any],
    ) -> float:
        features = features or {}

        info_mean = self._coerce_float(features.get("mean_information_bound", 0.0))
        info_max = self._coerce_float(features.get("max_information_bound", info_mean))
        info_score = (info_mean + info_max) / 2.0 if info_max else info_mean

        field_information = self._normalise_mapping(features.get("field_information", {}))
        candidate_field_info = {}
        metadata_field_info = None
        if isinstance(candidate.metadata, Mapping):
            metadata_field_info = candidate.metadata.get("field_information")
        if isinstance(metadata_field_info, Mapping):
            candidate_field_info = self._normalise_mapping(metadata_field_info)

        field_scores: List[float] = []
        for field in candidate.fields:
            value = candidate_field_info.get(field)
            if value is None:
                value = field_information.get(field)
            if value is None and info_mean:
                value = info_mean
            if value is not None:
                field_scores.append(self._coerce_float(value))
        field_score = sum(field_scores) / len(field_scores) if field_scores else info_score

        utilisation_samples: List[float] = []
        for observation in observations:
            if observation.tokens_total > 0:
                utilisation_samples.append(
                    float(observation.tokens_retained) / float(observation.tokens_total)
                )
            elif observation.compression_ratio:
                utilisation_samples.append(float(observation.compression_ratio))
        if not utilisation_samples:
            mean_ratio = self._coerce_float(features.get("mean_compression_ratio", 0.0))
            if mean_ratio:
                utilisation_samples.append(mean_ratio)
        utilisation = sum(utilisation_samples) / len(utilisation_samples) if utilisation_samples else 0.0

        tokens_retained_mean = self._coerce_float(features.get("mean_tokens_retained", 0.0))
        budget_mean = self._coerce_float(features.get("budget_mean", 0.0))
        estimated_need = tokens_retained_mean or budget_mean
        if not estimated_need:
            for observation in observations:
                estimated_need = max(estimated_need, float(observation.tokens_retained))
        if not estimated_need:
            estimated_need = float(candidate.budget)

        difference_ratio = 0.0
        if estimated_need > 0:
            difference_ratio = abs(float(candidate.budget) - estimated_need) / max(estimated_need, 1.0)
        balance = 1.0 - min(difference_ratio, 1.0)
        balance = max(self.min_balance, balance)

        info_strength = self.information_weight * info_score
        field_strength = self.field_weight * field_score
        utilisation_strength = self.utilisation_weight * utilisation

        history_factor = 1.0 + self.history_weight * (min(len(history), 10) / 10.0)
        diversity_factor = 1.0
        if candidate.fields:
            diversity_factor += self.diversity_weight * math.log1p(len(candidate.fields))

        pressure = max(0.0, utilisation - 0.75)
        pressure_factor = 1.0 + self.pressure_weight * pressure

        score_core = info_strength + field_strength + utilisation_strength
        if score_core <= 0:
            score_core = utilisation_strength + len(history) * 0.05 + len(candidate.fields) * 0.01

        budget = max(float(candidate.budget), 1e-3)

        score = score_core * history_factor * diversity_factor * pressure_factor * balance * budget
        return float(score)

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _normalise_mapping(self, value: Any) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            return {}
        result: Dict[str, float] = {}
        for key, raw in value.items():
            try:
                result[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        return result

@dataclass
class CompressionRecord:
    """Snapshot of compression outputs and telemetry destined for persistence."""

    compressed_fields: Mapping[str, Sequence[Any]]
    telemetry: Mapping[str, Any]
    metrics: Mapping[str, float]
    bottleneck: Optional[str] = None
    policy_metadata: Optional[Mapping[str, Any]] = None
    probe_outcomes: Sequence[Mapping[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        selected = self.telemetry.get("selected_indices", {})
        token_counts = self.telemetry.get("token_counts", {})

        total_tokens = 0
        if isinstance(token_counts, Mapping):
            total_tokens = sum(int(v) for v in token_counts.values())

        retained = 0
        if isinstance(selected, Mapping):
            for value in selected.values():
                retained += len(_iterable_to_list(value))

        ratio = float(retained) / float(total_tokens) if total_tokens else 0.0
        dropped = float(total_tokens - retained) if total_tokens else 0.0
        regenerated = dropped
        residual_stats = self.telemetry.get("residual_statistics", {})
        if isinstance(residual_stats, Mapping):
            regenerated_total = 0.0
            for stats in residual_stats.values():
                if isinstance(stats, Mapping):
                    try:
                        regenerated_total += float(stats.get("dropped_count", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
            if regenerated_total:
                regenerated = float(regenerated_total)
        summary: Dict[str, float] = {
            "tokens_total": float(total_tokens),
            "tokens_retained": float(retained),
            "tokens_dropped": float(dropped),
            "tokens_regenerated": float(regenerated),
            "compression_ratio": ratio,
        }
        if isinstance(self.metrics, Mapping):
            for name, value in self.metrics.items():
                try:
                    summary[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
        return summary

    def as_metadata(self) -> Dict[str, Any]:
        serialised_fields: Dict[str, List[Any]] = {
            str(field): [
                _to_serialisable(item)
                for item in _iterable_to_list(sequence)
            ]
            for field, sequence in self.compressed_fields.items()
        }
        telemetry_map = self.telemetry if isinstance(self.telemetry, Mapping) else {}
        metrics_map = {
            str(name): float(value)
            for name, value in self.metrics.items()
        }
        summary = self.summary()

        field_counts = {
            field: len(entries)
            for field, entries in serialised_fields.items()
        }
        pipeline: Dict[str, Any] = {}
        if self.bottleneck is not None:
            pipeline["bottleneck"] = str(self.bottleneck)

        idx_cells = self._build_idx_cells(telemetry_map, serialised_fields)
        artifacts = self._collect_artifacts(telemetry_map, serialised_fields)

        metadata: Dict[str, Any] = {
            "compressed_fields": serialised_fields,
            "telemetry": _to_serialisable(self.telemetry),
            "metrics": metrics_map,
            "summary": summary,
            "pipeline": pipeline,
            "fields": {
                "names": sorted(serialised_fields.keys()),
                "counts": field_counts,
            },
            "K": int(float(summary.get("tokens_retained", 0.0) or 0.0)),
            "mi_lb": self._resolve_mi_lower_bound(),
            "idx_cells": idx_cells,
            "artifacts": artifacts,
        }
        if self.bottleneck is not None:
            metadata["bottleneck"] = str(self.bottleneck)
        if self.policy_metadata:
            metadata["policy_metadata"] = _to_serialisable(self.policy_metadata)
        if self.probe_outcomes:
            metadata["probe_outcomes"] = [
                _to_serialisable(outcome) for outcome in self.probe_outcomes
            ]
        return metadata

    def _resolve_mi_lower_bound(self) -> Optional[float]:
        for source in (self.metrics, self.telemetry):
            if not isinstance(source, Mapping):
                continue
            for key in ("mi_lb", "mi_lower_bound", "mutual_information_lb"):
                value = source.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _build_idx_cells(
        self,
        telemetry: Mapping[str, Any],
        fields: Mapping[str, Sequence[Any]],
    ) -> Dict[str, Any]:
        idx_cells: Dict[str, Any] = {}

        kept = self._normalise_index_map(telemetry.get("selected_indices"))
        if kept:
            idx_cells["kept"] = kept

        dropped = self._normalise_index_map(telemetry.get("dropped_indices"))
        if dropped:
            idx_cells["dropped"] = dropped

        token_counts_raw = telemetry.get("token_counts")
        if isinstance(token_counts_raw, Mapping):
            token_counts = {}
            for field, value in token_counts_raw.items():
                try:
                    token_counts[str(field)] = int(value)
                except (TypeError, ValueError):
                    continue
            if token_counts:
                idx_cells["counts"] = token_counts

        canonical_ids = self._extract_canonical_cell_ids(fields)
        if canonical_ids:
            idx_cells["canonical"] = canonical_ids

        return idx_cells

    def _normalise_index_map(self, value: Any) -> Dict[str, List[int]]:
        if not isinstance(value, Mapping):
            return {}
        normalised: Dict[str, List[int]] = {}
        for field, indices in value.items():
            normalised_indices: List[int] = []
            for raw in _iterable_to_list(indices):
                try:
                    normalised_indices.append(int(raw))
                except (TypeError, ValueError):
                    continue
            if normalised_indices:
                normalised[str(field)] = normalised_indices
        return normalised

    def _extract_canonical_cell_ids(
        self,
        fields: Mapping[str, Sequence[Any]],
    ) -> Dict[str, List[str]]:
        canonical: Dict[str, List[str]] = {}
        for field, entries in fields.items():
            ids: List[str] = []
            for entry in entries:
                identifier = self._resolve_canonical_identifier(entry)
                if identifier is not None:
                    ids.append(str(identifier))
            if ids:
                canonical[str(field)] = ids
        return canonical

    def _resolve_canonical_identifier(self, entry: Any) -> Optional[Any]:
        if isinstance(entry, Mapping):
            for key in (
                "canonical_cell_id",
                "canonical_id",
                "cell_id",
                "cell_index",
                "canonical_cell",
                "cell",
                "id",
            ):
                if key not in entry:
                    continue
                value = entry[key]
                if isinstance(value, Mapping):
                    for nested_key in ("id", "cell_id", "index"):
                        if nested_key in value:
                            return value[nested_key]
                    continue
                if value is not None:
                    return value
            for nested in entry.values():
                identifier = self._resolve_canonical_identifier(nested)
                if identifier is not None:
                    return identifier
        if isinstance(entry, (list, tuple)):
            for item in entry:
                identifier = self._resolve_canonical_identifier(item)
                if identifier is not None:
                    return identifier
        return None

    def _collect_artifacts(
        self,
        telemetry: Mapping[str, Any],
        fields: Mapping[str, Sequence[Any]],
    ) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}
        for key in ("artifacts", "cell_artifacts", "rasterized_cells", "canonical_cells"):
            value = telemetry.get(key)
            if value is not None:
                artifacts[str(key)] = _to_serialisable(value)

        aggregated = self._aggregate_canonical_cells(telemetry, fields)
        if aggregated is not None and "canonical_cells" not in artifacts:
            artifacts["canonical_cells"] = aggregated

        return artifacts

    def _aggregate_canonical_cells(
        self,
        telemetry: Mapping[str, Any],
        fields: Mapping[str, Sequence[Any]],
    ) -> Optional[Any]:
        cell_centers = self._prepare_cell_centers(
            telemetry.get("cell_centers")
            or telemetry.get("canonical_cell_centers")
            or self._extract_artifact_centers(telemetry.get("artifacts"))
        )
        if not cell_centers:
            return None

        field_batches: List[Dict[str, Any]] = []
        for entries in fields.values():
            tokens_batch: List[List[float]] = []
            coords_batch: List[List[float]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                embedding = self._extract_embedding(entry)
                coords = self._extract_coordinates(entry)
                if embedding is None or coords is None:
                    continue
                tokens_batch.append(embedding)
                coords_batch.append(coords)
            if tokens_batch and len(tokens_batch) == len(coords_batch):
                field_batches.append({
                    "tokens": [tokens_batch],
                    "coords": [coords_batch],
                })

        if not field_batches:
            return None

        agg_mode = telemetry.get("cell_agg", "mean")
        tau = telemetry.get("cell_tau")
        backend = telemetry.get("cell_backend")
        try:
            from nd_llm.utils import aggregate_fields
        except Exception:  # pragma: no cover - optional dependency failures
            return None

        kwargs: Dict[str, Any] = {}
        if isinstance(agg_mode, str):
            kwargs["agg"] = agg_mode
        if isinstance(tau, (int, float)):
            kwargs["tau"] = float(tau)
        if isinstance(backend, str):
            kwargs["backend"] = backend

        try:
            aggregated = aggregate_fields(field_batches, cell_centers, **kwargs)
        except Exception:  # pragma: no cover - aggregation is best-effort
            return None
        return _to_serialisable(aggregated)

    def _extract_artifact_centers(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            for key in ("cell_centers", "centers", "canonical_centers"):
                if key in value:
                    return value[key]
        return None

    def _extract_embedding(self, entry: Mapping[str, Any]) -> Optional[List[float]]:
        for key in ("embedding", "vector", "tokens", "values"):
            if key in entry:
                vector = _iterable_to_list(entry[key])
                if vector:
                    try:
                        return [float(component) for component in vector]
                    except (TypeError, ValueError):
                        continue
        return None

    def _extract_coordinates(self, entry: Mapping[str, Any]) -> Optional[List[float]]:
        for key in ("coords", "coord", "position", "center", "centroid"):
            if key in entry:
                vector = _iterable_to_list(entry[key])
                if vector:
                    try:
                        return [float(component) for component in vector]
                    except (TypeError, ValueError):
                        continue
        return None

    def _prepare_cell_centers(self, value: Any) -> List[Any]:
        centres = _iterable_to_list(value)
        if not centres:
            return []
        if centres and isinstance(centres[0], (int, float)):
            return [[[_coerce_float(component) for component in centres]]]
        if centres and isinstance(centres[0], list):
            if centres[0] and isinstance(centres[0][0], (int, float)):
                return [[[_coerce_float(component) for component in centre] for centre in centres]]
            normalised: List[Any] = []
            for batch in centres:
                batch_list = _iterable_to_list(batch)
                normalised.append([
                    [_coerce_float(component) for component in centre]
                    for centre in batch_list
                    if isinstance(centre, (list, tuple))
                ])
            return normalised
        return []

    def to_compression_result(self) -> CompressionResult:
        telemetry = self.telemetry if isinstance(self.telemetry, Mapping) else {}
        selected_indices_raw = telemetry.get("selected_indices", {})
        selected_scores_raw = telemetry.get("selected_scores", {})
        token_counts_raw = telemetry.get("token_counts", {})
        budget_raw = telemetry.get("budget")

        selected_indices: Dict[str, List[int]] = {}
        if isinstance(selected_indices_raw, Mapping):
            for field, indices in selected_indices_raw.items():
                selected_indices[str(field)] = [_coerce_int(i) for i in _iterable_to_list(indices)]

        selected_scores: Dict[str, List[float]] = {}
        if isinstance(selected_scores_raw, Mapping):
            for field, scores in selected_scores_raw.items():
                selected_scores[str(field)] = [_coerce_float(score) for score in _iterable_to_list(scores)]

        token_counts: Dict[str, int] = {}
        if isinstance(token_counts_raw, Mapping):
            for field, count in token_counts_raw.items():
                token_counts[str(field)] = _coerce_int(count)

        budget = int(budget_raw) if isinstance(budget_raw, (int, float)) else sum(token_counts.values())

        field_budgets_raw = telemetry.get("field_budgets", {})
        field_budgets: Dict[str, int] = {}
        if isinstance(field_budgets_raw, Mapping):
            for field, value in field_budgets_raw.items():
                field_budgets[str(field)] = _coerce_int(value)

        allocation_weights_raw = telemetry.get("allocation_weights", {})
        allocation_weights: Dict[str, float] = {}
        if isinstance(allocation_weights_raw, Mapping):
            for field, value in allocation_weights_raw.items():
                allocation_weights[str(field)] = _coerce_float(value)

        dropped_indices_raw = telemetry.get("dropped_indices", {})
        dropped_indices: Dict[str, List[int]] = {}
        if isinstance(dropped_indices_raw, Mapping):
            for field, indices in dropped_indices_raw.items():
                dropped_indices[str(field)] = [_coerce_int(i) for i in _iterable_to_list(indices)]

        residual_stats_raw = telemetry.get("residual_statistics", {})
        residual_statistics: Dict[str, Dict[str, float]] = {}
        if isinstance(residual_stats_raw, Mapping):
            for field, stats in residual_stats_raw.items():
                if isinstance(stats, Mapping):
                    residual_statistics[str(field)] = {
                        str(name): _coerce_float(value)
                        for name, value in stats.items()
                    }

        quantized_embeddings_raw = telemetry.get("quantized_embeddings", {})
        quantized_embeddings: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(quantized_embeddings_raw, Mapping):
            for field, entries in quantized_embeddings_raw.items():
                normalised_entries: List[Dict[str, Any]] = []
                for entry in _iterable_to_list(entries):
                    if not isinstance(entry, Mapping):
                        continue
                    values_raw = entry.get("values", [])
                    if isinstance(values_raw, Mapping):
                        values_iterable = list(values_raw.values())
                    elif isinstance(values_raw, (list, tuple, set)):
                        values_iterable = list(values_raw)
                    else:
                        values_iterable = [values_raw]
                    values = [_coerce_int(val) for val in values_iterable]
                    scale = _coerce_float(entry.get("scale", 1.0))
                    if scale == 0.0:
                        scale = 1.0
                    normalised_entries.append(
                        {
                            "index": _coerce_int(entry.get("index", len(normalised_entries))),
                            "values": values,
                            "scale": scale,
                        }
                    )
                quantized_embeddings[str(field)] = normalised_entries

        telemetry_obj = CompressionTelemetry(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            token_counts=token_counts,
            budget=budget,
            field_budgets=field_budgets,
            allocation_weights=allocation_weights,
            dropped_indices=dropped_indices,
            residual_statistics=residual_statistics,
            quantized_embeddings=quantized_embeddings,
        )

        return CompressionResult(
            compressed_fields={
                str(field): [item for item in sequence]
                for field, sequence in self.compressed_fields.items()
            },
            telemetry=telemetry_obj,
            metrics={str(name): float(value) for name, value in self.metrics.items()},
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "CompressionRecord":
        compressed_fields = metadata.get("compressed_fields", {})
        telemetry = metadata.get("telemetry", {})
        metrics = metadata.get("metrics", {})
        bottleneck = metadata.get("bottleneck")
        policy_metadata_raw = metadata.get("policy_metadata")
        probe_outcomes_raw = metadata.get("probe_outcomes", [])

        def _ensure_mapping(value: Any) -> Dict[str, Any]:
            if isinstance(value, Mapping):
                return {str(k): v for k, v in value.items()}
            return {}

        def _ensure_sequence_mapping(value: Any) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            for entry in _iterable_to_list(value):
                if isinstance(entry, Mapping):
                    items.append({str(k): v for k, v in entry.items()})
            return items

        policy_metadata_dict = _ensure_mapping(policy_metadata_raw)
        policy_metadata: Optional[Dict[str, Any]]
        if policy_metadata_dict:
            policy_metadata = policy_metadata_dict
        else:
            policy_metadata = None

        return cls(
            compressed_fields={
                str(field): _iterable_to_list(sequence)
                for field, sequence in _ensure_mapping(compressed_fields).items()
            },
            telemetry=_ensure_mapping(telemetry),
            metrics={str(name): float(val) for name, val in _ensure_mapping(metrics).items()},
            bottleneck=str(bottleneck) if bottleneck is not None else None,
            policy_metadata=policy_metadata,
            probe_outcomes=_ensure_sequence_mapping(probe_outcomes_raw),
        )

    @classmethod
    def from_result(
        cls,
        result: CompressionResult,
        *,
        bottleneck: Optional[Union[str, IBottleneck]] = None,
        policy_metadata: Optional[Mapping[str, Any]] = None,
        probe_outcomes: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "CompressionRecord":
        """Create a record directly from a :class:`CompressionResult`."""

        telemetry = result.telemetry
        if isinstance(bottleneck, IBottleneck):
            bottleneck_name: Optional[str] = bottleneck.__class__.__name__
        else:
            bottleneck_name = str(bottleneck) if bottleneck is not None else None

        telemetry_payload: Dict[str, Any] = {
            "selected_indices": telemetry.selected_indices,
            "selected_scores": telemetry.selected_scores,
            "token_counts": telemetry.token_counts,
            "budget": telemetry.budget,
            "field_budgets": getattr(telemetry, "field_budgets", {}),
            "allocation_weights": getattr(telemetry, "allocation_weights", {}),
            "dropped_indices": getattr(telemetry, "dropped_indices", {}),
            "residual_statistics": getattr(telemetry, "residual_statistics", {}),
            "quantized_embeddings": getattr(telemetry, "quantized_embeddings", {}),
        }

        return cls(
            compressed_fields=result.compressed_fields,
            telemetry=telemetry_payload,
            metrics=result.metrics,
            bottleneck=bottleneck_name,
            policy_metadata={str(k): v for k, v in dict(policy_metadata or {}).items()} or None,
            probe_outcomes=[
                {str(k): v for k, v in dict(outcome).items()}
                for outcome in list(probe_outcomes or [])
                if isinstance(outcome, Mapping)
            ],
        )


@dataclass
class UsageEvent:
    """Container describing a single usage event destined for persistence."""

    tensor: TensorLike
    metadata: Optional[Mapping[str, Any]] = None
    key: Optional[str] = None
    compression: Optional[CompressionRecord] = None


Snapshot = Tuple[str, "CompressionRecord", Mapping[str, Any]]


class Orchestrator:
    """Co-ordinates persistence of usage events and lightweight policy probes."""

    def __init__(
        self,
        stm: STM,
        config: OrchestratorConfig,
        bottleneck: Optional["IBottleneck"] = None,
        meta_model: Optional[BudgetMetaModel] = None,
        *,
        auto_attach_meta_model: bool = True,
        constraints: Optional[Sequence[ConstraintModule]] = None,
        superposition_channels: Optional[Sequence[str]] = None,
    ) -> None:
        self._stm = stm
        self._config = config
        self._bottleneck = bottleneck
        if meta_model is None and auto_attach_meta_model:
            meta_model = HeuristicBudgetMetaModel()
        self._meta_model = meta_model
        self._usage_log: List[str] = []
        self._budget_history: List[BudgetDecision] = []
        self._last_policy_metadata: Optional[Dict[str, Any]] = None
        self._recent_probe_outcomes: List[Dict[str, Any]] = []
        self._probe_history_limit = 5
        self._constraints: List[ConstraintModule] = list(constraints or [])
        self._recent_constraint_results: List[Dict[str, Any]] = []
        self._superposition_channels: Tuple[str, ...] = tuple(
            channel for channel in (superposition_channels or []) if channel
        )

    @classmethod
    def from_components(
        cls,
        *,
        target_budget: float,
        policy_name: str = "default",
        budget_step: float = 0.1,
        retention_probe_sample_size: int = 10,
        stm: Optional[STM] = None,
        storage_dir: Optional[Union[str, Path]] = None,
        index_filename: str = "index.json",
        bottleneck: Optional["IBottleneck"] = None,
        meta_model: Optional[BudgetMetaModel] = None,
        auto_attach_meta_model: bool = True,
        constraints: Optional[Sequence[ConstraintModule]] = None,
        superposition_channels: Optional[Sequence[str]] = None,
    ) -> "Orchestrator":
        """Construct an orchestrator from primitive components."""

        if stm is None:
            if storage_dir is None:
                raise ValueError(
                    "from_components requires either an existing STM or a storage_dir"
                )
            stm = STM.from_path(storage_dir, index_filename=index_filename)

        config = OrchestratorConfig(
            target_budget=target_budget,
            policy_name=policy_name,
            budget_step=budget_step,
            retention_probe_sample_size=retention_probe_sample_size,
        )
        return cls(
            stm=stm,
            config=config,
            bottleneck=bottleneck,
            meta_model=meta_model,
            auto_attach_meta_model=auto_attach_meta_model,
            constraints=constraints,
            superposition_channels=superposition_channels,
        )

    @property
    def config(self) -> OrchestratorConfig:
        return self._config

    @property
    def usage_log(self) -> List[str]:
        return list(self._usage_log)

    @property
    def bottleneck(self) -> Optional["IBottleneck"]:
        """Return the bottleneck instance associated with the orchestrator."""

        return self._bottleneck

    @bottleneck.setter
    def bottleneck(self, value: Optional["IBottleneck"]) -> None:
        self._bottleneck = value

    @property
    def meta_model(self) -> Optional[BudgetMetaModel]:
        """Return the meta-model attached to the orchestrator, if any."""

        return self._meta_model

    @meta_model.setter
    def meta_model(self, value: Optional[BudgetMetaModel]) -> None:
        self._meta_model = value

    def log_usage_event(self, event: UsageEvent) -> str:
        """Persist an event's tensor payload and metadata via the STM."""

        base_key = event.key or self._generate_key()
        metadata = dict(event.metadata or {})
        metadata.setdefault("policy_name", self._config.policy_name)
        metadata.setdefault("target_budget", self._config.target_budget)
        metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        if event.compression is not None:
            compression_metadata = event.compression.as_metadata()
            existing_compression = metadata.get("compression")
            if isinstance(existing_compression, Mapping):
                merged_compression = dict(existing_compression)
                merged_compression.update(compression_metadata)
            else:
                merged_compression = dict(compression_metadata)

            summary = merged_compression.get("summary", {})
            compression_record = event.compression
            if self._last_policy_metadata and not compression_record.policy_metadata:
                compression_record = replace(
                    compression_record,
                    policy_metadata=dict(self._last_policy_metadata),
                )

            if self._recent_probe_outcomes:
                existing = [
                    dict(outcome)
                    for outcome in _iterable_to_list(compression_record.probe_outcomes)
                    if isinstance(outcome, Mapping)
                ]
                seen_ids = {
                    str(outcome.get("id"))
                    for outcome in existing
                    if isinstance(outcome, Mapping)
                }
                combined = existing[:]
                for outcome in self._recent_probe_outcomes:
                    outcome_id = str(outcome.get("id")) if outcome.get("id") is not None else None
                    if outcome_id is None or outcome_id not in seen_ids:
                        combined.append(dict(outcome))
                        if outcome_id is not None:
                            seen_ids.add(outcome_id)
                if self._probe_history_limit > 0:
                    combined = combined[-self._probe_history_limit :]
                compression_record = replace(
                    compression_record,
                    probe_outcomes=combined,
                )

            event.compression = compression_record
            compression_metadata = compression_record.as_metadata()
            merged_compression.update(compression_metadata)
            metadata["compression"] = merged_compression
            summary = merged_compression.get("summary", {})
            if isinstance(summary, Mapping):
                for field in (
                    "compression_ratio",
                    "tokens_retained",
                    "tokens_total",
                    "tokens_regenerated",
                    "tokens_dropped",
                ):
                    value = summary.get(field)
                    if value is not None and field not in metadata:
                        metadata[field] = value
            telemetry = merged_compression.get("telemetry", {})
            if isinstance(telemetry, Mapping) and "budget" in telemetry and "compression_budget" not in metadata:
                metadata["compression_budget"] = telemetry["budget"]

            idx_cells = merged_compression.get("idx_cells", {})
            if isinstance(idx_cells, Mapping) and "idx_cells" not in metadata:
                metadata["idx_cells"] = idx_cells

            compression_artifacts = merged_compression.get("artifacts")
            if isinstance(compression_artifacts, Mapping):
                existing_artifacts = metadata.get("artifacts")
                merged_artifacts = dict(compression_artifacts)
                if isinstance(existing_artifacts, Mapping):
                    merged_artifacts.update(existing_artifacts)
                metadata["artifacts"] = merged_artifacts

            if "K" not in metadata and merged_compression.get("K") is not None:
                metadata["K"] = merged_compression["K"]

            mi_lb = merged_compression.get("mi_lb")
            if mi_lb is not None and "mi_lb" not in metadata:
                metadata["mi_lb"] = mi_lb

            layout_signature = _compute_layout_signature(idx_cells)
            if layout_signature and "layout_signature" not in metadata:
                metadata["layout_signature"] = layout_signature
            if layout_signature and "layout_signature" not in merged_compression:
                merged_compression["layout_signature"] = layout_signature

        constraint_results = self._evaluate_constraints(event)
        if constraint_results:
            metadata["constraints"] = [result.to_dict() for result in constraint_results]
            if any(not result.satisfied for result in constraint_results):
                issues = metadata.setdefault("issues", [])
                for result in constraint_results:
                    if not result.satisfied:
                        issues.append(
                            {
                                "type": "constraint_violation",
                                "constraint": result.name,
                                "details": dict(result.details),
                            }
                        )

        metadata.setdefault("task", metadata.get("policy_name", self._config.policy_name))

        attempt_key = base_key
        duplicate_attempts = 0
        while True:
            try:
                self._stm.append(attempt_key, event.tensor, metadata)
                break
            except KeyError:
                duplicate_attempts += 1
                metadata = dict(metadata)
                metadata.setdefault("duplicate_of", base_key)
                metadata["duplicate_attempts"] = duplicate_attempts
                attempt_key = self._generate_key(prefix=base_key)

        self._update_superpositions(event, metadata)
        self._usage_log.append(attempt_key)
        return attempt_key

    def budget_sweep(self, budget_values: Optional[Iterable[float]] = None) -> Dict[float, Dict[str, Any]]:
        """Perform a simple sweep across potential budget values."""

        if budget_values is None:
            start = max(0.0, self._config.target_budget - self._config.budget_step)
            budget_values = (
                start,
                self._config.target_budget,
                self._config.target_budget + self._config.budget_step,
            )

        results: Dict[float, Dict[str, Any]] = {}
        for raw_budget in budget_values:
            budget = float(raw_budget)
            delta = budget - self._config.target_budget
            results[budget] = {
                "budget": budget,
                "delta_from_target": delta,
                "within_budget": delta <= 0,
            }
        return results

    def tune_budget(
        self,
        *,
        window: Optional[int] = None,
        strategy: Optional[BudgetStrategy] = None,
    ) -> float:
        """Adjust the orchestrator budget based on recent telemetry observations."""

        snapshots = self._collect_record_snapshots(window=window)
        observations = self._collect_budget_observations(window=window, snapshots=snapshots)
        history_features = self._build_history_features(snapshots, observations)
        if strategy is None:
            strategy = CompressionRatioBudgetStrategy(step=self._config.budget_step)

        decision = strategy.propose(self._config.target_budget, observations)
        decision = replace(
            decision,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                **decision.metadata,
                "observations": [obs.as_dict() for obs in observations],
                "history_features": history_features,
            },
        )

        meta_model = self._meta_model
        meta_summary: Optional[Dict[str, Any]] = None
        if meta_model and snapshots:
            field_sets = self._infer_candidate_field_sets(snapshots)
            candidate_budgets = self._candidate_budgets(decision.proposed_budget, snapshots)
            history_records = [record for _, record, _ in snapshots]
            evaluations: List[Dict[str, Any]] = []
            best_evaluation: Optional[Dict[str, Any]] = None

            for fields in field_sets:
                for budget in candidate_budgets:
                    candidate_metadata = {
                        "history_record_count": history_features.get("record_count", 0),
                        "mean_information_bound": history_features.get("mean_information_bound", 0.0),
                        "field_information": {
                            field: history_features.get("field_information", {}).get(field, 0.0)
                            for field in fields
                        },
                    }
                    candidate = BudgetCandidate(fields=fields, budget=budget, metadata=candidate_metadata)

                    try:
                        score = float(
                            meta_model.score_candidate(
                                candidate,
                                history=history_records,
                                observations=observations,
                                features=history_features,
                            )
                        )
                    except Exception:
                        score = float(self._fallback_candidate_score(candidate, snapshots))

                    evaluation = {"candidate": candidate.as_dict(), "score": score}
                    evaluations.append(evaluation)
                    if best_evaluation is None or score > best_evaluation.get("score", float("-inf")):
                        best_evaluation = evaluation

            if evaluations:
                model_name = getattr(meta_model, "name", meta_model.__class__.__name__)
                meta_summary = {
                    "model": model_name,
                    "evaluations": evaluations,
                    "features": history_features,
                    "candidate_space": {
                        "budgets": candidate_budgets,
                        "field_sets": [list(field_set) for field_set in field_sets],
                    },
                }
                if best_evaluation is not None:
                    meta_summary["selected"] = best_evaluation
                    selected_budget = float(best_evaluation["candidate"]["budget"])
                    if abs(selected_budget - float(decision.proposed_budget)) > 1e-6:
                        decision = replace(
                            decision,
                            proposed_budget=selected_budget,
                            reason=f"meta-model:{model_name}",
                        )

                decision = replace(
                    decision,
                    metadata={**decision.metadata, "meta_model": meta_summary},
                )

        self._budget_history.append(decision)

        if decision.proposed_budget != self._config.target_budget:
            self._config = replace(self._config, target_budget=decision.proposed_budget)

        self._last_policy_metadata = {
            "policy": self._config.policy_name,
            "decision": decision.as_dict(),
            "snapshot_keys": [key for key, _, _ in snapshots],
        }
        if meta_summary is not None:
            self._last_policy_metadata["meta_model"] = meta_summary

        return decision.proposed_budget

    def run_retention_probe(self) -> Dict[str, Any]:
        """Run a placeholder retention probe over STM contents."""

        keys = list(self._stm.list_keys())
        sample_size = min(len(keys), self._config.retention_probe_sample_size)
        sample = keys[-sample_size:]

        qualities: List[float] = []
        retained_ratios: List[float] = []
        regenerated_ratios: List[float] = []
        mse_values: List[float] = []
        kl_values: List[float] = []
        issues: List[Dict[str, Any]] = []
        probe_entries: List[str] = []
        reconstruction_details: List[Dict[str, Any]] = []

        for key in sample:
            try:
                index_entry = self._stm.get_index_entry(key)
            except KeyError:
                issues.append({"key": key, "issue": "missing_index_entry"})
                continue

            metadata = index_entry.get("metadata", {})
            compression_data = metadata.get("compression") if isinstance(metadata, Mapping) else None
            if not isinstance(compression_data, Mapping):
                issues.append({"key": key, "issue": "missing_compression_metadata"})
                continue

            record = CompressionRecord.from_metadata(compression_data)
            compression_result = record.to_compression_result()
            budget = compression_result.telemetry.budget or int(self._config.target_budget)
            budget = max(int(budget), 1)
            bottleneck = IBottleneck(target_budget=budget)
            reconstructed = bottleneck.decompress(compression_result)

            metrics = reconstructed.get("metrics", {}) if isinstance(reconstructed, Mapping) else {}
            retained_ratio = float(metrics.get("retained_ratio", 0.0) or 0.0)
            regenerated_ratio = float(metrics.get("regenerated_ratio", 0.0) or 0.0)
            mean_mse = float(metrics.get("mean_mse", 0.0) or 0.0)
            mean_kl = float(metrics.get("mean_kl_divergence", 0.0) or 0.0)

            retained_ratios.append(retained_ratio)
            regenerated_ratios.append(regenerated_ratio)
            mse_values.append(mean_mse)
            kl_values.append(mean_kl)

            quality = max(0.0, min(1.0, retained_ratio * (1.0 - min(mean_mse, 1.0))))
            qualities.append(quality)
            probe_entries.append(key)

            summary = record.summary()
            reconstruction_details.append(
                {
                    "key": key,
                    "retained_ratio": retained_ratio,
                    "regenerated_ratio": regenerated_ratio,
                    "mean_mse": mean_mse,
                    "mean_kl_divergence": mean_kl,
                    "quality": quality,
                    "tokens_total": summary.get("tokens_total", 0.0),
                }
            )

        reconstruction_summary: Dict[str, Any]
        if qualities:
            def _mean(values: Sequence[float]) -> float:
                return float(sum(values) / len(values)) if values else 0.0

            reconstruction_summary = {
                "mean_quality": _mean(qualities),
                "min_quality": float(min(qualities)),
                "max_quality": float(max(qualities)),
                "mean_retained_ratio": _mean(retained_ratios),
                "mean_regenerated_ratio": _mean(regenerated_ratios),
                "mean_mse": _mean(mse_values),
                "mean_kl_divergence": _mean(kl_values),
                "sample_size": len(qualities),
                "details": reconstruction_details,
            }
        else:
            reconstruction_summary = {
                "mean_quality": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
                "mean_retained_ratio": 0.0,
                "mean_regenerated_ratio": 0.0,
                "mean_mse": 0.0,
                "mean_kl_divergence": 0.0,
                "sample_size": 0,
                "details": [],
            }

        result = {
            "total_retained": len(keys),
            "sampled_keys": probe_entries,
            "log_size": len(self._usage_log),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reconstruction": reconstruction_summary,
            "issues": issues,
        }

        meta_context = self._resolve_meta_model_probe_context()
        result["meta_model"] = meta_context

        self._append_probe_outcome(
            {
                "type": "retention_probe",
                "timestamp": result["timestamp"],
                "summary": reconstruction_summary,
                "issues": issues,
                "sampled_keys": probe_entries,
                "meta_model": meta_context,
            }
        )

        return result

    def run_proxy_trial(
        self,
        *,
        candidate_budget: float,
        fields: Optional[Sequence[str]] = None,
        window: Optional[int] = None,
        include_adversarial: bool = False,
        adversarial_limit: int = 5,
    ) -> Dict[str, Any]:
        """Score a candidate policy using stored compression telemetry."""

        snapshots = self._collect_record_snapshots(window=window)
        observations = self._collect_budget_observations(window=window, snapshots=snapshots)
        features = self._build_history_features(snapshots, observations)

        field_sets = self._infer_candidate_field_sets(snapshots)
        if fields is None and field_sets:
            fields = list(field_sets[0])

        candidate_fields = tuple(str(field) for field in (fields or ()))
        candidate_metadata = {
            "history_record_count": features.get("record_count", 0),
            "mean_information_bound": features.get("mean_information_bound", 0.0),
            "field_information": {
                field: features.get("field_information", {}).get(field, 0.0)
                for field in candidate_fields
            },
        }
        candidate = BudgetCandidate(
            fields=candidate_fields,
            budget=float(candidate_budget),
            metadata=candidate_metadata,
        )

        meta_model = self._meta_model
        if meta_model:
            history_records = [record for _, record, _ in snapshots]
            try:
                score = float(
                    meta_model.score_candidate(
                        candidate,
                        history=history_records,
                        observations=observations,
                        features=features,
                    )
                )
            except Exception:
                score = float(self._fallback_candidate_score(candidate, snapshots))
        else:
            score = float(self._fallback_candidate_score(candidate, snapshots))

        timestamp = datetime.now(timezone.utc).isoformat()
        result: Dict[str, Any] = {
            "candidate": candidate.as_dict(),
            "score": score,
            "history_window": len(snapshots),
            "timestamp": timestamp,
            "features": features,
            "sample_keys": [key for key, _, _ in snapshots],
        }

        meta_context = self._resolve_meta_model_probe_context(
            candidate=candidate, score=score
        )
        result["meta_model"] = meta_context

        if include_adversarial:
            result["adversarial_samples"] = self.generate_adversarial_samples(
                limit=adversarial_limit,
                window=window,
                snapshots=snapshots,
            )

        self._append_probe_outcome(
            {
                "type": "proxy_trial",
                "timestamp": timestamp,
                "candidate": result["candidate"],
                "score": score,
                "meta_model": meta_context,
            }
        )

        return result

    def generate_adversarial_samples(
        self,
        *,
        limit: int = 5,
        window: Optional[int] = None,
        snapshots: Optional[Sequence[Snapshot]] = None,
    ) -> List[Dict[str, Any]]:
        """Return high-severity compression samples for rapid evaluation."""

        if snapshots is None:
            snapshots = self._collect_record_snapshots(window=window)

        samples: List[Dict[str, Any]] = []
        for key, record, _ in snapshots:
            summary = record.summary()
            metrics = dict(record.metrics)
            mutual_information = float(metrics.get("information_bound", 0.0) or 0.0)
            tokens_total = float(summary.get("tokens_total", 0.0) or 0.0)
            tokens_retained = float(summary.get("tokens_retained", 0.0) or 0.0)
            tokens_dropped = max(tokens_total - tokens_retained, 0.0)
            dropped_ratio = float(tokens_dropped / tokens_total) if tokens_total else 0.0
            info_shortfall = max(0.0, 1.0 - min(mutual_information, 1.0))
            severity = dropped_ratio + info_shortfall

            telemetry = record.telemetry if isinstance(record.telemetry, Mapping) else {}
            fields = list(record.compressed_fields.keys())
            if not fields:
                token_counts = telemetry.get("token_counts", {}) if isinstance(telemetry, Mapping) else {}
                if isinstance(token_counts, Mapping):
                    fields = [str(field) for field in token_counts.keys()]

            samples.append(
                {
                    "key": key,
                    "fields": sorted(str(field) for field in fields),
                    "summary": summary,
                    "mutual_information": mutual_information,
                    "telemetry": _to_serialisable(telemetry),
                    "policy_metadata": _to_serialisable(record.policy_metadata)
                    if record.policy_metadata
                    else None,
                    "probe_outcomes": [
                        _to_serialisable(outcome) for outcome in record.probe_outcomes
                    ],
                    "severity": severity,
                }
            )

        samples.sort(key=lambda item: float(item.get("severity", 0.0)), reverse=True)
        if limit > 0:
            samples = samples[:limit]

        return samples

    def _append_probe_outcome(self, outcome: Mapping[str, Any]) -> None:
        payload = {str(k): _to_serialisable(v) for k, v in dict(outcome).items()}
        payload.setdefault("id", uuid.uuid4().hex[:12])
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._recent_probe_outcomes.append(payload)
        if self._probe_history_limit > 0 and len(self._recent_probe_outcomes) > self._probe_history_limit:
            self._recent_probe_outcomes = self._recent_probe_outcomes[-self._probe_history_limit :]

    def _evaluate_constraints(self, event: UsageEvent) -> List[ConstraintResult]:
        if not self._constraints:
            return []
        results: List[ConstraintResult] = []
        for module in self._constraints:
            try:
                outcome = module.evaluate(
                    stm=self._stm,
                    event=event,
                    compression=event.compression,
                )
            except Exception as exc:
                outcome = ConstraintResult(
                    name=getattr(module, "name", module.__class__.__name__),
                    satisfied=False,
                    confidence=0.0,
                    details={"error": str(exc)},
                )
            results.append(outcome)
        if results:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": [result.to_dict() for result in results],
            }
            self._recent_constraint_results.append(record)
            if self._probe_history_limit > 0:
                self._recent_constraint_results = self._recent_constraint_results[-self._probe_history_limit :]
        return results

    def _update_superpositions(self, event: UsageEvent, metadata: Mapping[str, Any]) -> None:
        if not self._superposition_channels:
            return
        payload = event.tensor
        if payload is None:
            return
        base_metadata = {
            "policy_name": metadata.get("policy_name"),
            "task": metadata.get("task"),
        }
        for channel in self._superposition_channels:
            try:
                self._stm.write_superposition(channel, payload, metadata=base_metadata)
            except Exception:
                continue

    def _resolve_meta_model_probe_context(
        self,
        *,
        candidate: Optional[BudgetCandidate] = None,
        score: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        meta_model = self._meta_model
        model_name = (
            getattr(meta_model, "name", meta_model.__class__.__name__)
            if meta_model
            else None
        )

        summary: Optional[Mapping[str, Any]] = None
        if self._budget_history:
            latest = self._budget_history[-1].metadata.get("meta_model")
            if isinstance(latest, Mapping):
                summary = latest
        if summary is None and self._last_policy_metadata:
            fallback = self._last_policy_metadata.get("meta_model")
            if isinstance(fallback, Mapping):
                summary = fallback

        if summary is None and model_name is None:
            return None

        context: Dict[str, Any] = {}
        if model_name is not None:
            context["model"] = model_name

        if summary is not None:
            selected = summary.get("selected")
            if isinstance(selected, Mapping):
                context["selected"] = {
                    "candidate": _to_serialisable(selected.get("candidate")),
                    "score": selected.get("score"),
                }

        if candidate is not None:
            context["candidate"] = candidate.as_dict()
        if score is not None:
            context["score"] = float(score)

        return context or None

    def _collect_record_snapshots(self, *, window: Optional[int]) -> List[Snapshot]:
        if window is None:
            window = min(len(self._usage_log), self._config.retention_probe_sample_size)

        recent_keys = self._usage_log[-window:]
        snapshots: List[Snapshot] = []

        for key in reversed(recent_keys):
            try:
                entry = self._stm.get_index_entry(key)
            except KeyError:
                continue

            metadata = entry.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue

            compression_data = metadata.get("compression")
            if not isinstance(compression_data, Mapping):
                continue

            record = CompressionRecord.from_metadata(compression_data)
            snapshots.append((key, record, metadata))

        return snapshots

    def _build_history_features(
        self,
        snapshots: Sequence[Snapshot],
        observations: Sequence[BudgetObservation],
    ) -> Dict[str, Any]:
        def _safe_mean(values: Sequence[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        info_bounds: List[float] = []
        field_information: Dict[str, List[float]] = {}
        budgets: List[float] = []

        for _, record, _ in snapshots:
            metrics = record.metrics if isinstance(record.metrics, Mapping) else {}
            info_value = metrics.get("information_bound")
            if isinstance(info_value, (int, float)):
                info_bounds.append(float(info_value))

            telemetry = record.telemetry if isinstance(record.telemetry, Mapping) else {}
            field_mi = telemetry.get("field_mutual_information", {})
            if isinstance(field_mi, Mapping):
                for field, value in field_mi.items():
                    try:
                        field_information.setdefault(str(field), []).append(float(value))
                    except (TypeError, ValueError):
                        continue

            budget_value = telemetry.get("budget")
            if isinstance(budget_value, (int, float)):
                budgets.append(float(budget_value))

        mean_ratio = _safe_mean([obs.compression_ratio for obs in observations])
        mean_tokens_total = _safe_mean([float(obs.tokens_total) for obs in observations])
        mean_tokens_retained = _safe_mean([float(obs.tokens_retained) for obs in observations])

        features: Dict[str, Any] = {
            "mean_information_bound": _safe_mean(info_bounds),
            "max_information_bound": max(info_bounds) if info_bounds else 0.0,
            "field_information": {
                field: _safe_mean(values)
                for field, values in field_information.items()
            },
            "mean_compression_ratio": mean_ratio,
            "mean_tokens_total": mean_tokens_total,
            "mean_tokens_retained": mean_tokens_retained,
            "observation_count": len(observations),
            "record_count": len(snapshots),
            "budget_mean": _safe_mean(budgets),
            "record_summaries": [record.summary() for _, record, _ in snapshots],
            "recent_fields": [
                sorted(
                    {
                        str(field)
                        for field in (
                            list(record.compressed_fields.keys())
                            or list(
                                (record.telemetry or {}).get("token_counts", {}).keys()
                                if isinstance(record.telemetry, Mapping)
                                else []
                            )
                        )
                    }
                )
                for _, record, _ in snapshots
            ],
        }

        return features

    def _infer_candidate_field_sets(self, snapshots: Sequence[Snapshot]) -> List[Tuple[str, ...]]:
        field_sets: List[Tuple[str, ...]] = []
        union_fields: List[str] = []

        for _, record, _ in snapshots:
            fields = [str(field) for field in record.compressed_fields.keys()]
            telemetry = record.telemetry if isinstance(record.telemetry, Mapping) else {}
            if not fields and isinstance(telemetry, Mapping):
                token_counts = telemetry.get("token_counts", {})
                if isinstance(token_counts, Mapping):
                    fields = [str(field) for field in token_counts.keys()]

            if fields:
                sorted_fields = tuple(sorted(fields))
                if sorted_fields not in field_sets:
                    field_sets.append(sorted_fields)
                for field in sorted_fields:
                    if field not in union_fields:
                        union_fields.append(field)

        if union_fields:
            union_tuple = tuple(sorted(union_fields))
            if union_tuple not in field_sets:
                field_sets.append(union_tuple)

        if not field_sets:
            field_sets.append(tuple())

        return field_sets

    def _candidate_budgets(self, base_budget: float, snapshots: Sequence[Snapshot]) -> List[float]:
        budgets: List[float] = [float(base_budget), float(self._config.target_budget)]
        step = max(float(self._config.budget_step), 0.0)

        if step > 0:
            budgets.append(max(step, float(self._config.target_budget) - step))
            budgets.append(float(self._config.target_budget) + step)

        for _, record, _ in snapshots:
            telemetry = record.telemetry if isinstance(record.telemetry, Mapping) else {}
            budget_value = telemetry.get("budget")
            if isinstance(budget_value, (int, float)):
                budgets.append(float(budget_value))
            field_budgets = telemetry.get("field_budgets")
            if isinstance(field_budgets, Mapping):
                for value in field_budgets.values():
                    if isinstance(value, (int, float)):
                        budgets.append(float(value))

        min_budget = max(step or 1e-3, 1e-3)
        normalised = {
            round(float(budget) if budget > 0 else min_budget, 6)
            for budget in budgets
        }

        return sorted(normalised)

    def _fallback_candidate_score(
        self,
        candidate: BudgetCandidate,
        snapshots: Sequence[Snapshot],
    ) -> float:
        if not snapshots:
            return 0.0

        mi_values: List[float] = []
        drop_ratios: List[float] = []

        for _, record, _ in snapshots:
            metrics = record.metrics if isinstance(record.metrics, Mapping) else {}
            info_value = metrics.get("information_bound")
            if isinstance(info_value, (int, float)):
                mi_values.append(float(info_value))

            summary = record.summary()
            tokens_total = float(summary.get("tokens_total", 0.0) or 0.0)
            tokens_retained = float(summary.get("tokens_retained", 0.0) or 0.0)
            if tokens_total > 0:
                drop_ratios.append(max(0.0, 1.0 - (tokens_retained / tokens_total)))

        mean_mi = float(sum(mi_values) / len(mi_values)) if mi_values else 0.0
        mean_drop = float(sum(drop_ratios) / len(drop_ratios)) if drop_ratios else 0.0
        denominator = max(candidate.budget, 1e-3)

        return (mean_mi + 1.0 - mean_drop) / denominator

    def _generate_key(self, prefix: Optional[str] = None) -> str:
        base = prefix or "event"
        safe_base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
        return f"{safe_base}-{uuid.uuid4().hex[:8]}"

    def _collect_budget_observations(
        self,
        *,
        window: Optional[int],
        snapshots: Optional[Sequence[Snapshot]] = None,
    ) -> List[BudgetObservation]:
        if snapshots is None:
            snapshots = self._collect_record_snapshots(window=window)

        observations: List[BudgetObservation] = []

        for key, record, metadata in snapshots:
            summary = record.summary()
            compression_ratio = float(summary.get("compression_ratio", 0.0) or 0.0)
            tokens_total = int(summary.get("tokens_total", 0.0) or 0)
            tokens_retained = int(summary.get("tokens_retained", 0.0) or 0)
            timestamp_value = metadata.get("timestamp") if isinstance(metadata, Mapping) else None
            timestamp = timestamp_value if isinstance(timestamp_value, str) else None

            observations.append(
                BudgetObservation(
                    key=key,
                    compression_ratio=compression_ratio,
                    tokens_total=tokens_total,
                    tokens_retained=tokens_retained,
                    metrics=dict(record.metrics),
                    timestamp=timestamp,
                )
            )

        return observations

    @property
    def budget_history(self) -> List[BudgetDecision]:
        return list(self._budget_history)


__all__ = [
    "BudgetCandidate",
    "BudgetMetaModel",
    "HeuristicBudgetMetaModel",
    "CompressionRecord",
    "Orchestrator",
    "UsageEvent",
]
