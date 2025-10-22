"""Orchestration utilities coordinating STM persistence and policy sweeps."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from nd_llm.bottleneck.ib import CompressionResult, CompressionTelemetry, IBottleneck
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
class CompressionRecord:
    """Snapshot of compression outputs and telemetry destined for persistence."""

    compressed_fields: Mapping[str, Sequence[Any]]
    telemetry: Mapping[str, Any]
    metrics: Mapping[str, float]
    bottleneck: Optional[str] = None

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

        def _ensure_mapping(value: Any) -> Dict[str, Any]:
            if isinstance(value, Mapping):
                return {str(k): v for k, v in value.items()}
            return {}

        return cls(
            compressed_fields={
                str(field): _iterable_to_list(sequence)
                for field, sequence in _ensure_mapping(compressed_fields).items()
            },
            telemetry=_ensure_mapping(telemetry),
            metrics={str(name): float(val) for name, val in _ensure_mapping(metrics).items()},
            bottleneck=str(bottleneck) if bottleneck is not None else None,
        )

    @classmethod
    def from_result(
        cls,
        result: CompressionResult,
        *,
        bottleneck: Optional[Union[str, IBottleneck]] = None,
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
        )


@dataclass
class UsageEvent:
    """Container describing a single usage event destined for persistence."""

    tensor: TensorLike
    metadata: Optional[Mapping[str, Any]] = None
    key: Optional[str] = None
    compression: Optional[CompressionRecord] = None


class Orchestrator:
    """Co-ordinates persistence of usage events and lightweight policy probes."""

    def __init__(
        self,
        stm: STM,
        config: OrchestratorConfig,
        bottleneck: Optional["IBottleneck"] = None,
    ) -> None:
        self._stm = stm
        self._config = config
        self._bottleneck = bottleneck
        self._usage_log: List[str] = []
        self._budget_history: List[BudgetDecision] = []

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
        return cls(stm=stm, config=config, bottleneck=bottleneck)

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
                merged_compression = compression_metadata
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

        observations = self._collect_budget_observations(window=window)
        if strategy is None:
            strategy = CompressionRatioBudgetStrategy(step=self._config.budget_step)

        decision = strategy.propose(self._config.target_budget, observations)
        decision = replace(
            decision,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={**decision.metadata, "observations": [obs.as_dict() for obs in observations]},
        )
        self._budget_history.append(decision)

        if decision.proposed_budget != self._config.target_budget:
            self._config = replace(self._config, target_budget=decision.proposed_budget)

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

        return {
            "total_retained": len(keys),
            "sampled_keys": probe_entries,
            "log_size": len(self._usage_log),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reconstruction": reconstruction_summary,
            "issues": issues,
        }

    def _generate_key(self, prefix: Optional[str] = None) -> str:
        base = prefix or "event"
        safe_base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
        return f"{safe_base}-{uuid.uuid4().hex[:8]}"

    def _collect_budget_observations(self, *, window: Optional[int]) -> List[BudgetObservation]:
        if window is None:
            window = min(len(self._usage_log), self._config.retention_probe_sample_size)

        recent_keys = self._usage_log[-window:]
        observations: List[BudgetObservation] = []

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
            summary = record.summary()
            compression_ratio = float(summary.get("compression_ratio", 0.0) or 0.0)
            tokens_total = int(summary.get("tokens_total", 0.0) or 0)
            tokens_retained = int(summary.get("tokens_retained", 0.0) or 0)
            timestamp = metadata.get("timestamp") if isinstance(metadata.get("timestamp"), str) else None

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


__all__ = ["CompressionRecord", "Orchestrator", "UsageEvent"]
