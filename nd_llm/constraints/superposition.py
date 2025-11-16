"""Constraints that operate on STM holographic superpositions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from .base import ConstraintModule, ConstraintResult


def _flatten_tensor(payload: Any) -> List[float]:
    if payload is None:
        return []
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, (list, tuple, Iterable)):
        result: List[float] = []
        for item in payload:
            result.extend(_flatten_tensor(item))
        return result
    return []


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class SuperpositionSimilarityConstraint(ConstraintModule):
    """Ensure the current tensor is aligned with the historical superposition."""

    channel: str
    min_similarity: float = 0.2
    name: str = "superposition_similarity"

    def evaluate(
        self,
        *,
        stm: "STM",  # type: ignore[name-defined]  # pragma: no cover - typing hook
        event: "UsageEvent",  # type: ignore[name-defined]
        compression: Optional["CompressionRecord"],  # type: ignore[name-defined]
    ) -> ConstraintResult:
        try:
            historical, metadata = stm.read_superposition(self.channel, normalize=True)
        except KeyError:
            return ConstraintResult(
                name=self.name,
                satisfied=True,
                details={"channel": self.channel, "reason": "empty_channel"},
            )
        current = _flatten_tensor(event.tensor)
        if len(current) != len(historical):
            return ConstraintResult(
                name=self.name,
                satisfied=True,
                details={"channel": self.channel, "reason": "shape_mismatch"},
            )
        similarity = _cosine_similarity(current, historical)
        satisfied = similarity >= float(self.min_similarity)
        details = {
            "channel": self.channel,
            "similarity": float(similarity),
            "threshold": float(self.min_similarity),
            "weight": float(metadata.get("weight", 0.0) or 0.0),
        }
        return ConstraintResult(
            name=self.name,
            satisfied=satisfied,
            confidence=max(0.0, min(1.0, similarity)),
            details=details,
        )
