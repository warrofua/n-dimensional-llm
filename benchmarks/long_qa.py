"""Synthetic long-context QA benchmark aligning with ND-LLM scaffolds."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from nd_llm.bottleneck import CompressionResult
from nd_llm.encoders import Encoder, LayoutEncoder, TextEncoder
from nd_llm.registry import Registry

from .doc_understanding import (
    AblationFn,
    BudgetRun,
    _drop_field_ablation,
    _evaluate_budget,
    _noise_field_ablation,
    _perturb_layout_ablation,
    _shuffle_field_ablation,
)


@dataclass(frozen=True)
class LongQASegment:
    dialog_id: int
    turn_id: int
    text: str
    keyword: str
    start: float
    end: float
    salience: float


@dataclass(frozen=True)
class LongQADialog:
    dialog_id: int
    question: str
    answer_turn: int
    answer_keyword: str
    segments: Sequence[LongQASegment]

    def as_fields(self) -> Dict[str, List[MutableMapping[str, Any]]]:
        return longqa_fields(
            {
                "dialog_id": self.dialog_id,
                "question": self.question,
                "answer_turn": self.answer_turn,
                "answer_keyword": self.answer_keyword,
                "segments": [
                    {
                        "dialog_id": segment.dialog_id,
                        "turn_id": segment.turn_id,
                        "text": segment.text,
                        "keyword": segment.keyword,
                        "start": segment.start,
                        "end": segment.end,
                        "salience": segment.salience,
                    }
                    for segment in self.segments
                ],
            }
        )


_TOPICS = [
    "arctic expedition",
    "lunar colony",
    "solar observatory",
    "deep sea research",
    "volcanic monitoring",
]

_FACT_TOKENS = [
    "temperature",
    "habitat",
    "sample",
    "anomaly",
    "calibration",
]


def build_longqa_registry() -> Registry:
    registry = Registry()
    registry.add_field("question", keys=["dialog_id"], salience=True, modality="text")
    registry.add_field("context", keys=["dialog_id", "turn_id"], salience=True, modality="text")
    registry.add_field("layout", keys=["dialog_id", "turn_id"], modality="layout")
    registry.add_field("timeline", keys=["dialog_id", "turn_id"], modality="temporal")
    registry.add_affinity("context", "layout", keys=["dialog_id", "turn_id"])
    registry.add_affinity("context", "timeline", keys=["dialog_id", "turn_id"])
    registry.validate()
    return registry


def build_longqa_encoders(
    registry: Registry,
    *,
    question_dim: int = 12,
    context_dim: int = 12,
    timeline_dim: int = 4,
    layout_dim: int = 4,
) -> Dict[str, Encoder]:
    encoders: Dict[str, Encoder] = {
        "question": TextEncoder(embedding_dim=question_dim),
        "context": TextEncoder(embedding_dim=context_dim),
        "layout": LayoutEncoder(embedding_dim=layout_dim),
        "timeline": LayoutEncoder(embedding_dim=timeline_dim),
    }
    for name, encoder in encoders.items():
        registry.register_encoder(name, encoder)
    return encoders


def synthetic_longqa_dialog(
    dialog_id: int,
    *,
    num_turns: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed * 101 + dialog_id * 53)
    topic = rng.choice(_TOPICS)
    answer_turn = rng.randrange(max(num_turns, 1))
    answer_keyword = f"clue-{dialog_id}-{answer_turn}"
    segments: List[Dict[str, Any]] = []
    for turn_id in range(num_turns):
        base_token = rng.choice(_FACT_TOKENS)
        text = f"Turn {turn_id}: {topic} report on {base_token}"
        if turn_id == answer_turn:
            text += f" revealing {answer_keyword}"
            keyword = answer_keyword
            salience = 0.95
        else:
            keyword = f"{base_token}-{turn_id}"
            salience = 0.35 + 0.15 * rng.random()
        start = float(turn_id) * 1.0
        end = start + 0.8
        segments.append(
            {
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "text": text,
                "keyword": keyword,
                "start": start,
                "end": end,
                "salience": salience,
            }
        )
    question = f"Which turn mentions {answer_keyword}?"
    return {
        "dialog_id": dialog_id,
        "question": question,
        "answer_turn": answer_turn,
        "answer_keyword": answer_keyword,
        "segments": segments,
    }


def synthetic_longqa_dataset(
    count: int,
    *,
    seed: int = 0,
    num_turns: int = 6,
) -> List[Dict[str, Any]]:
    return [
        synthetic_longqa_dialog(dialog_id=index + 1, num_turns=num_turns, seed=seed)
        for index in range(count)
    ]


def longqa_fields(document: Mapping[str, Any]) -> Dict[str, List[MutableMapping[str, Any]]]:
    dialog_id = int(document.get("dialog_id", 0))
    segments = list(document.get("segments", []))
    total_duration = max(
        (float(segment.get("end", 0.0)) for segment in segments),
        default=float(len(segments)),
    )
    total_duration = total_duration if total_duration > 0 else float(len(segments) or 1)
    fields: Dict[str, List[MutableMapping[str, Any]]] = {
        "question": [
            {
                "dialog_id": dialog_id,
                "text": str(document.get("question", "")),
            }
        ],
        "context": [],
        "layout": [],
        "timeline": [],
    }
    for segment in segments:
        turn_id = int(segment.get("turn_id", 0))
        start = float(segment.get("start", turn_id))
        end = float(segment.get("end", start + 0.8))
        span = max(end - start, 1e-6)
        start_norm = min(max(start / total_duration, 0.0), 1.0)
        end_norm = min(max((start + span) / total_duration, 0.0), 1.0)
        keyword = str(segment.get("keyword", ""))
        fields["context"].append(
            {
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "text": str(segment.get("text", "")),
                "keyword": keyword,
                "salience": float(segment.get("salience", 0.5)),
            }
        )
        bbox = [start_norm, 0.1, max(start_norm, end_norm), 0.9]
        fields["layout"].append(
            {
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "xyxy": bbox,
            }
        )
        fields["timeline"].append(
            {
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "xyxy": [start_norm, 0.0, max(start_norm, end_norm), 1.0],
                "duration": span,
            }
        )
    return fields


def _longqa_label(document: Mapping[str, Any]) -> int:
    return int(document.get("answer_turn", -1))


def _longqa_predict(result: CompressionResult, document: Mapping[str, Any]) -> int:
    keyword = document.get("answer_keyword")
    context_field = result.compressed_fields.get("context", [])
    for entry in context_field:
        if not isinstance(entry, Mapping):
            continue
        text = str(entry.get("text", ""))
        if keyword and keyword in text:
            return int(entry.get("turn_id", -1))
    return -1


def _longqa_metadata(document: Mapping[str, Any], result: Any) -> Mapping[str, Any]:
    return {
        "dialog_id": document.get("dialog_id"),
        "answer_turn": document.get("answer_turn"),
        "kept_fields": {k: len(v) for k, v in getattr(result, "compressed_fields", {}).items()},
    }


def _longqa_ablation_suite() -> Dict[str, AblationFn]:
    return {
        "drop_timeline": _drop_field_ablation("timeline"),
        "drop_layout": _drop_field_ablation("layout"),
        "shuffle_context": _shuffle_field_ablation("context"),
        "perturb_timeline": _perturb_layout_ablation(scale=0.05, field="timeline"),
        "noise_duration": _noise_field_ablation("timeline", "duration", scale=0.2),
    }


def run_long_qa_benchmark(
    budget_values: Sequence[int] = (6, 10, 14),
    *,
    dataset_size: int = 12,
    seed: int = 0,
    num_turns: int = 6,
) -> Dict[str, Any]:
    registry = build_longqa_registry()
    encoders = build_longqa_encoders(registry)
    dataset = synthetic_longqa_dataset(dataset_size, seed=seed, num_turns=num_turns)
    ablations = _longqa_ablation_suite()

    runs: List[BudgetRun] = []
    for budget in budget_values:
        result = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            fields_fn=longqa_fields,
            label_fn=_longqa_label,
            predict_fn=_longqa_predict,
            metadata_fn=_longqa_metadata,
            policy_name="long-qa-benchmark",
            retention_probe_sample_size=4,
            seed=seed,
            ablations=ablations,
        )
        runs.append(result)

    return {
        "dataset": "synthetic-long-qa",
        "dataset_size": len(dataset),
        "num_turns": num_turns,
        "budgets": [run.to_dict() for run in runs],
    }


__all__ = [
    "LongQADialog",
    "LongQASegment",
    "build_longqa_registry",
    "build_longqa_encoders",
    "synthetic_longqa_dialog",
    "synthetic_longqa_dataset",
    "longqa_fields",
    "run_long_qa_benchmark",
]
