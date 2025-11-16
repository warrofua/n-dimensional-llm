"""ChartQA benchmark utilities leveraging the N-D registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from nd_llm.encoders import Encoder, TextEncoder
from nd_llm.registry import Registry
from nd_llm.utils import build_mi_proxy_context
from nd_llm.bottleneck import IBottleneck

try:  # pragma: no cover - optional HF dependency
    from datasets import load_dataset as _load_hf_dataset  # type: ignore
except Exception:  # pragma: no cover
    _load_hf_dataset = None  # type: ignore[assignment]

_DATA_DIR = Path(__file__).with_name("data")
_SAMPLE_PATH = _DATA_DIR.joinpath("chartqa_sample.jsonl")
_DATASET_NAME = "lmms-lab/chartqa"

__all__ = [
    "load_chartqa_dataset",
    "build_chartqa_registry",
    "build_chartqa_encoders",
    "chartqa_fields",
    "chartqa_answer",
    "run_chartqa_benchmark",
]


def load_chartqa_dataset(
    *,
    split: str = "test",
    limit: Optional[int] = None,
    use_sample: bool = True,
    cache_dir: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Load ChartQA records either from the bundled sample or Hugging Face."""

    documents: List[Dict[str, Any]] = []
    if use_sample:
        documents.extend(_load_chartqa_sample(limit))
        if documents:
            return documents
    if _load_hf_dataset is None:
        raise ImportError(
            "datasets is required to fetch ChartQA from Hugging Face. Install it or set use_sample=True."
        )
    dataset = _load_hf_dataset(
        _DATASET_NAME,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    for index, record in enumerate(dataset):
        document = _prepare_chartqa_document(record, f"{split}-{index}")
        documents.append(document)
        if limit is not None and len(documents) >= limit:
            break
    return documents


def _load_chartqa_sample(limit: Optional[int]) -> List[Dict[str, Any]]:
    if not _SAMPLE_PATH.exists():
        return []
    docs: List[Dict[str, Any]] = []
    with _SAMPLE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            docs.append(json.loads(line))
            if limit is not None and len(docs) >= limit:
                break
    return docs


def _prepare_chartqa_document(raw: Mapping[str, Any], default_doc_id: str) -> Dict[str, Any]:
    doc_id = str(raw.get("id") or raw.get("doc_id") or f"chartqa-{default_doc_id}")
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    chart_data = raw.get("chart") or raw.get("chart_data") or []
    parsed_chart: List[Dict[str, Any]] = []
    for entry in chart_data:
        if not isinstance(entry, Mapping):
            continue
        label = entry.get("label") or entry.get("category") or entry.get("x")
        value = entry.get("value") or entry.get("y")
        if label is None or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        parsed_chart.append({"label": str(label), "value": numeric})
    metadata = {
        "chart_type": raw.get("type"),
        "source": str(raw.get("source", "")),
    }
    return {
        "doc_id": doc_id,
        "question": question,
        "answer": answer,
        "chart": parsed_chart,
        "metadata": metadata,
    }


def build_chartqa_registry() -> Registry:
    registry = Registry()
    registry.add_field("question", keys=["doc_id", "token_id"], salience=True, modality="text")
    registry.add_field("chart", keys=["doc_id", "row_id"], modality="table")
    registry.add_affinity("question", "chart", keys=["doc_id"])
    registry.validate()
    return registry


def build_chartqa_encoders(registry: Registry, *, question_dim: int = 16, chart_dim: int = 12) -> Dict[str, Encoder]:
    encoders: Dict[str, Encoder] = {
        "question": TextEncoder(embedding_dim=question_dim),
        "chart": TextEncoder(embedding_dim=chart_dim),
    }
    for field, encoder in encoders.items():
        registry.register_encoder(field, encoder)
    return encoders


def chartqa_fields(document: Mapping[str, Any]) -> Dict[str, List[MutableMapping[str, Any]]]:
    doc_id = str(document.get("doc_id") or "")
    question = str(document.get("question", ""))
    chart_entries = list(document.get("chart") or [])
    question_tokens: List[MutableMapping[str, Any]] = [
        {
            "doc_id": doc_id,
            "token_id": idx,
            "text": token,
        }
        for idx, token in enumerate(question.split())
    ]
    chart_field: List[MutableMapping[str, Any]] = []
    for row_id, entry in enumerate(chart_entries):
        label = entry.get("label")
        value = entry.get("value")
        if label is None or value is None:
            continue
        chart_entry: MutableMapping[str, Any] = {
            "doc_id": doc_id,
            "row_id": row_id,
            "label": str(label),
            "value": float(value),
        }
        chart_field.append(chart_entry)
    return {"question": question_tokens, "chart": chart_field}


def chartqa_answer(document: Mapping[str, Any]) -> str:
    return str(document.get("answer", "")).strip()


def run_chartqa_benchmark(
    budget_values: Iterable[int] = (4, 8),
    *,
    dataset_size: int = 4,
    split: str = "test",
    use_sample: bool = True,
    cache_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    registry = build_chartqa_registry()
    build_chartqa_encoders(registry)
    limit = dataset_size if dataset_size > 0 else None
    dataset = load_chartqa_dataset(split=split, limit=limit, use_sample=use_sample, cache_dir=cache_dir)
    actual_size = len(dataset)

    budgets: List[Dict[str, Any]] = []
    for budget in budget_values:
        metrics = _evaluate_chartqa_budget(
            dataset=dataset,
            budget=int(budget),
            registry_encoders=registry.encoders,
        )
        budgets.append(metrics)

    return {
        "dataset": "ChartQA",
        "split": split,
        "dataset_size": actual_size,
        "use_sample": bool(use_sample),
        "budgets": budgets,
    }


def _evaluate_chartqa_budget(
    *,
    dataset: Sequence[Mapping[str, Any]],
    budget: int,
    registry_encoders: Mapping[str, Encoder],
) -> Dict[str, Any]:
    bottleneck = IBottleneck(target_budget=int(budget))
    correct = 0
    doc_count = 0
    info_bound = 0.0
    rate_dist = 0.0
    mi_total = 0.0
    mi_count = 0
    kept_totals = 0

    for document in dataset:
        fields = chartqa_fields(document)
        mi_proxy, mi_context = build_mi_proxy_context(
            fields,
            registry_encoders,
            preferred_fields=("question", "chart"),
        )
        result = bottleneck.compress(
            fields,
            encoders=registry_encoders,
            context=mi_context,
            mi_proxy=mi_proxy,
        )
        prediction = _chartqa_predict(document, result)
        if prediction == chartqa_answer(document):
            correct += 1
        metrics = result.metrics
        info_bound += float(metrics.get("information_bound", 0.0) or 0.0)
        rate_dist += float(metrics.get("rate_distortion", 0.0) or 0.0)
        mi_value = metrics.get("mi_lower_bound")
        if mi_value is not None:
            mi_total += float(mi_value)
            mi_count += 1
        kept = sum(len(indices) for indices in result.telemetry.selected_indices.values())
        kept_totals += kept
        doc_count += 1

    accuracy = float(correct) / float(doc_count or 1)
    return {
        "budget": int(budget),
        "accuracy": accuracy,
        "distortion": 1.0 - accuracy,
        "average_kept_tokens": kept_totals / float(doc_count or 1),
        "mean_information_bound": info_bound / float(doc_count or 1),
        "mean_rate_distortion": rate_dist / float(doc_count or 1),
        "mean_mi_lower_bound": mi_total / float(mi_count or 1) if mi_count else 0.0,
        "evaluated_documents": doc_count,
    }


def _chartqa_predict(document: Mapping[str, Any], result: Any) -> str:
    question = str(document.get("question", "")).lower()
    chart_entries = document.get("chart") or []
    if chart_entries and isinstance(chart_entries, list):
        try:
            values = [(entry["label"], float(entry["value"])) for entry in chart_entries]
        except Exception:
            values = []
    else:
        values = []

    if values:
        if "highest" in question or "most" in question or "maximum" in question:
            label = max(values, key=lambda item: item[1])[0]
            return str(label)
        if "lowest" in question or "least" in question or "minimum" in question:
            label = min(values, key=lambda item: item[1])[0]
            return str(label)
        if "total" in question or "sum" in question:
            total = sum(value for _, value in values)
            return str(int(total) if total.is_integer() else round(total, 2))
    compressed_questions = result.compressed_fields.get("question", [])
    if compressed_questions:
        first = compressed_questions[0]
        if isinstance(first, Mapping):
            return str(first.get("text", ""))
        return str(first)
    return ""
