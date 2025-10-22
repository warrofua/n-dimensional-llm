"""Orchestration utilities coordinating STM persistence and policy sweeps."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from nd_llm.stm import STM, TensorLike
from nd_llm.utils.config import OrchestratorConfig


@dataclass
class UsageEvent:
    """Container describing a single usage event destined for persistence."""

    tensor: TensorLike
    metadata: Optional[Mapping[str, Any]] = None
    key: Optional[str] = None


class Orchestrator:
    """Co-ordinates persistence of usage events and lightweight policy probes."""

    def __init__(self, stm: STM, config: OrchestratorConfig) -> None:
        self._stm = stm
        self._config = config
        self._usage_log: List[str] = []

    @property
    def config(self) -> OrchestratorConfig:
        return self._config

    @property
    def usage_log(self) -> List[str]:
        return list(self._usage_log)

    def log_usage_event(self, event: UsageEvent) -> str:
        """Persist an event's tensor payload and metadata via the STM."""

        base_key = event.key or self._generate_key()
        metadata = dict(event.metadata or {})
        metadata.setdefault("policy_name", self._config.policy_name)
        metadata.setdefault("target_budget", self._config.target_budget)
        metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

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

    def run_retention_probe(self) -> Dict[str, Any]:
        """Run a placeholder retention probe over STM contents."""

        keys = list(self._stm.list_keys())
        sample_size = min(len(keys), self._config.retention_probe_sample_size)
        sample = keys[:sample_size]
        return {
            "total_retained": len(keys),
            "sampled_keys": sample,
            "log_size": len(self._usage_log),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_key(self, prefix: Optional[str] = None) -> str:
        base = prefix or "event"
        safe_base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
        return f"{safe_base}-{uuid.uuid4().hex[:8]}"


__all__ = ["Orchestrator", "UsageEvent"]
