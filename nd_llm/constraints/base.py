"""Constraint interfaces used by the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

if False:  # pragma: no cover - typing aid
    from nd_llm.stm import STM
    from nd_llm.orchestration.orchestrator import CompressionRecord, UsageEvent


@dataclass
class ConstraintResult:
    """Evaluation outcome returned by constraint modules."""

    name: str
    satisfied: bool
    confidence: float = 1.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, Any]:
        payload = {
            "name": self.name,
            "satisfied": bool(self.satisfied),
            "confidence": float(self.confidence),
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@runtime_checkable
class ConstraintModule(Protocol):
    """Protocol describing orchestrator-aware constraints."""

    name: str

    def evaluate(
        self,
        *,
        stm: "STM",
        event: "UsageEvent",
        compression: Optional["CompressionRecord"],
    ) -> ConstraintResult:
        """Evaluate the constraint against the current usage event."""
