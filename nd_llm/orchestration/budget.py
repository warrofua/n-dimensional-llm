"""Budget adaptation strategies for the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass
class BudgetObservation:
    """Lightweight view of compression telemetry for budget tuning."""

    key: str
    compression_ratio: float
    tokens_total: int
    tokens_retained: int
    metrics: Mapping[str, Any]
    timestamp: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "compression_ratio": float(self.compression_ratio),
            "tokens_total": int(self.tokens_total),
            "tokens_retained": int(self.tokens_retained),
            "metrics": dict(self.metrics),
            "timestamp": self.timestamp,
        }


@dataclass
class BudgetDecision:
    """Decision emitted by a :class:`BudgetStrategy`."""

    proposed_budget: float
    reason: str
    utilisation: float
    adjustment: float
    telemetry_window: int
    metadata: Dict[str, Any]
    timestamp: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "proposed_budget": float(self.proposed_budget),
            "reason": self.reason,
            "utilisation": float(self.utilisation),
            "adjustment": float(self.adjustment),
            "telemetry_window": int(self.telemetry_window),
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }
        return payload


class BudgetStrategy:
    """Interface for budget adaptation policies."""

    def propose(
        self,
        current_budget: float,
        observations: Sequence[BudgetObservation],
    ) -> BudgetDecision:
        raise NotImplementedError


class CompressionRatioBudgetStrategy(BudgetStrategy):
    """Adjust budgets based on average compression utilisation."""

    def __init__(
        self,
        *,
        lower_bound: float = 0.55,
        upper_bound: float = 0.9,
        step: float = 0.1,
        min_budget: float = 0.1,
    ) -> None:
        if lower_bound <= 0 or upper_bound <= 0:
            raise ValueError("bounds must be positive")
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")
        if step <= 0:
            raise ValueError("step must be positive")
        if min_budget <= 0:
            raise ValueError("min_budget must be positive")

        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.step = float(step)
        self.min_budget = float(min_budget)

    def propose(
        self,
        current_budget: float,
        observations: Sequence[BudgetObservation],
    ) -> BudgetDecision:
        if not observations:
            return BudgetDecision(
                proposed_budget=float(current_budget),
                reason="steady",
                utilisation=0.0,
                adjustment=0.0,
                telemetry_window=0,
                metadata={"note": "no telemetry"},
            )

        ratios = [obs.compression_ratio for obs in observations if obs.tokens_total > 0]
        if not ratios:
            return BudgetDecision(
                proposed_budget=float(current_budget),
                reason="steady",
                utilisation=0.0,
                adjustment=0.0,
                telemetry_window=len(observations),
                metadata={"note": "missing ratios"},
            )

        avg_ratio = mean(ratios)
        adjustment = 0.0
        reason = "steady"

        if avg_ratio < self.lower_bound:
            adjustment = self.step
            reason = "under-utilised"
        elif avg_ratio > self.upper_bound:
            adjustment = -self.step
            reason = "over-utilised"

        proposed = max(self.min_budget, float(current_budget) + adjustment)
        metadata = {
            "average_ratio": avg_ratio,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "raw_ratios": ratios,
        }

        return BudgetDecision(
            proposed_budget=proposed,
            reason=reason,
            utilisation=avg_ratio,
            adjustment=adjustment,
            telemetry_window=len(ratios),
            metadata=metadata,
        )


__all__ = [
    "BudgetDecision",
    "BudgetObservation",
    "BudgetStrategy",
    "CompressionRatioBudgetStrategy",
]
