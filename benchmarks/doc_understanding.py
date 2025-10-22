"""Synthetic doc-understanding benchmark for accuracy vs. token budget."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from nd_llm.bottleneck import IBottleneck
from nd_llm.orchestration import Orchestrator, UsageEvent
from nd_llm.stm import STM
from nd_llm.utils import OrchestratorConfig, STMConfig

from .synthetic import (
    build_invoice_encoders,
    build_invoice_registry,
    high_value_label,
    invoice_fields,
    synthetic_invoice_dataset,
)


@dataclass
class BudgetRun:
    """Container summarising a single budget evaluation."""

    budget: int
    accuracy: float
    kept_tokens: List[int]
    budget_probe: Dict[float, Dict[str, Any]]
    retention_probe: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget,
            "accuracy": self.accuracy,
            "average_kept_tokens": sum(self.kept_tokens) / len(self.kept_tokens)
            if self.kept_tokens
            else 0.0,
            "budget_probe": self.budget_probe,
            "retention_probe": self.retention_probe,
        }


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
    for budget in budget_values:
        budget_run = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            threshold=threshold,
            seed=seed,
        )
        runs.append(budget_run)

    return {
        "dataset_size": dataset_size,
        "threshold": threshold,
        "budgets": [run.to_dict() for run in runs],
    }


def _evaluate_budget(
    *,
    budget: int,
    dataset: Sequence[Mapping[str, Any]],
    registry_encoders: Mapping[str, Any],
    threshold: float,
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
                policy_name="synthetic-doc-benchmark",
                budget_step=1.0,
                retention_probe_sample_size=5,
            ),
        )

        correct = 0
        kept_tokens: List[int] = []
        for invoice in dataset:
            fields = invoice_fields(invoice)
            result = bottleneck.compress(fields, encoders=registry_encoders)
            prediction = _predict_high_value(result.compressed_fields.get("amount", []), threshold)
            label = high_value_label(invoice, threshold)
            if prediction == label:
                correct += 1

            kept = sum(len(indices) for indices in result.telemetry.selected_indices.values())
            kept_tokens.append(kept)

            scores_vector = _scores_to_tensor(result.telemetry.selected_scores)
            orchestrator.log_usage_event(
                UsageEvent(
                    tensor=[scores_vector],
                    metadata={
                        "doc_id": invoice.get("doc_id"),
                        "budget": budget,
                        "kept_tokens": kept,
                        "label": label,
                        "prediction": prediction,
                    },
                )
            )

        accuracy = correct / len(dataset) if dataset else 0.0
        budget_probe = orchestrator.budget_sweep()
        retention_probe = orchestrator.run_retention_probe()

    return BudgetRun(
        budget=budget,
        accuracy=accuracy,
        kept_tokens=kept_tokens,
        budget_probe=budget_probe,
        retention_probe=retention_probe,
    )


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


def main() -> None:
    """CLI entry point printing a JSON benchmark report."""

    report = run_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["run_benchmark"]
