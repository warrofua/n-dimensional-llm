"""Orchestration utilities coordinating STM persistence and policy sweeps."""

from __future__ import annotations

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
        return {
            "tokens_total": float(total_tokens),
            "tokens_retained": float(retained),
            "compression_ratio": ratio,
        }

    def as_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "compressed_fields": {
                str(field): [
                    _to_serialisable(item)
                    for item in _iterable_to_list(sequence)
                ]
                for field, sequence in self.compressed_fields.items()
            },
            "telemetry": _to_serialisable(self.telemetry),
            "metrics": {
                str(name): float(value)
                for name, value in self.metrics.items()
            },
            "summary": self.summary(),
        }
        if self.bottleneck is not None:
            metadata["bottleneck"] = str(self.bottleneck)
        return metadata

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

        telemetry_obj = CompressionTelemetry(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            token_counts=token_counts,
            budget=budget,
            field_budgets=field_budgets,
            allocation_weights=allocation_weights,
            dropped_indices=dropped_indices,
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
            metadata.setdefault("compression", compression_metadata)
            summary = compression_metadata.get("summary", {})
            if isinstance(summary, Mapping):
                for field in ("compression_ratio", "tokens_retained", "tokens_total"):
                    value = summary.get(field)
                    if value is not None and field not in metadata:
                        metadata[field] = value
            telemetry = compression_metadata.get("telemetry", {})
            if isinstance(telemetry, Mapping) and "budget" in telemetry and "compression_budget" not in metadata:
                metadata["compression_budget"] = telemetry["budget"]

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
        issues: List[Dict[str, Any]] = []
        probe_entries: List[str] = []

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

            summary = record.summary()
            total_tokens = summary.get("tokens_total", 0.0) or 0.0
            retained = sum(len(values) for values in reconstructed.values())
            quality = float(retained) / float(total_tokens) if total_tokens else 0.0
            qualities.append(quality)
            probe_entries.append(key)

        reconstruction_summary: Dict[str, Any]
        if qualities:
            reconstruction_summary = {
                "mean_quality": float(sum(qualities) / len(qualities)),
                "min_quality": float(min(qualities)),
                "max_quality": float(max(qualities)),
                "sample_size": len(qualities),
            }
        else:
            reconstruction_summary = {
                "mean_quality": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
                "sample_size": 0,
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
