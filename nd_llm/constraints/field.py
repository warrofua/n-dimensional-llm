"""Field-level constraint modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .base import ConstraintModule, ConstraintResult


def _count_tokens(
    compression: Optional["CompressionRecord"],  # type: ignore[name-defined]
    field: str,
) -> int:
    if compression is None:
        return 0
    selected = compression.telemetry.get("selected_indices", {})
    if isinstance(selected, Mapping):
        tokens = selected.get(field)
        if isinstance(tokens, Mapping):
            return len(list(tokens.values()))
        if isinstance(tokens, (list, tuple)):
            return len(tokens)
    return 0


@dataclass
class FieldActivationConstraint(ConstraintModule):
    """Ensure a field keeps at least (and optionally at most) a number of tokens."""

    field: str
    min_tokens: int = 1
    max_tokens: Optional[int] = None
    name: str = "field_activation"

    def evaluate(
        self,
        *,
        stm: "STM",  # type: ignore[name-defined]  # pragma: no cover - protocol annotation
        event: "UsageEvent",  # type: ignore[name-defined]
        compression: Optional["CompressionRecord"],  # type: ignore[name-defined]
    ) -> ConstraintResult:
        count = _count_tokens(compression or event.compression, self.field)
        within_max = True
        if self.max_tokens is not None:
            within_max = count <= int(self.max_tokens)
        satisfied = count >= int(self.min_tokens) and within_max
        details: Mapping[str, Any] = {
            "field": self.field,
            "count": int(count),
            "min_tokens": int(self.min_tokens),
        }
        if self.max_tokens is not None:
            details = {**details, "max_tokens": int(self.max_tokens)}
        return ConstraintResult(
            name=self.name,
            satisfied=satisfied,
            confidence=1.0,
            details=details,
        )
