"""Synthetic doc-understanding benchmark for accuracy vs. token budget."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from nd_llm.bottleneck import CompressionResult, IBottleneck
from nd_llm.orchestration import Orchestrator, UsageEvent
from nd_llm.stm import STM
from nd_llm.utils import OrchestratorConfig, STMConfig

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


@dataclass
class BudgetRun:
    """Container summarising a single budget evaluation."""

    budget: int
    accuracy: float
    kept_tokens: List[int]
    budget_probe: Dict[float, Dict[str, Any]]
    retention_probe: Dict[str, Any]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget,
            "accuracy": self.accuracy,
            "average_kept_tokens": sum(self.kept_tokens) / len(self.kept_tokens)
            if self.kept_tokens
            else 0.0,
            "budget_probe": self.budget_probe,
            "retention_probe": self.retention_probe,
            "metrics": dict(self.metrics),
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
    predict_fn = _invoice_prediction_factory(threshold)
    for budget in budget_values:
        budget_run = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            fields_fn=invoice_fields,
            label_fn=lambda invoice, thr=threshold: high_value_label(invoice, thr),
            predict_fn=predict_fn,
            metadata_fn=_invoice_metadata,
            policy_name="synthetic-doc-benchmark",
            retention_probe_sample_size=5,
            seed=seed,
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
        )
        runs.append(budget_run)

    return {
        "dataset": "FUNSD",
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
) -> BudgetRun:
    bottleneck = IBottleneck(target_budget=int(budget))

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
        for document in dataset:
            fields = fields_fn(document)
            result = bottleneck.compress(fields, encoders=registry_encoders)
            prediction = predict_fn(result, document)
            label = label_fn(document)
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

        accuracy = correct / len(dataset) if dataset else 0.0
        metrics = {key: value / len(dataset) for key, value in metric_totals.items()} if dataset else {}
        budget_probe = orchestrator.budget_sweep()
        retention_probe = orchestrator.run_retention_probe()

    return BudgetRun(
        budget=budget,
        accuracy=accuracy,
        kept_tokens=kept_tokens,
        budget_probe=budget_probe,
        retention_probe=retention_probe,
        metrics=metrics,
    )


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


__all__ = ["run_benchmark", "run_funsd_benchmark"]
