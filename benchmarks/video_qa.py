"""Synthetic video QA benchmark covering temporal-visual fields."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

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
class VideoFrame:
    clip_id: int
    frame_id: int
    caption: str
    object: str
    bbox: Sequence[float]
    intensity: float


@dataclass(frozen=True)
class VideoClip:
    clip_id: int
    question: str
    target_object: str
    target_frame: int
    frames: Sequence[VideoFrame]

    def as_fields(self) -> Dict[str, List[MutableMapping[str, Any]]]:
        return videoqa_fields(
            {
                "clip_id": self.clip_id,
                "question": self.question,
                "target_object": self.target_object,
                "target_frame": self.target_frame,
                "frames": [
                    {
                        "clip_id": frame.clip_id,
                        "frame_id": frame.frame_id,
                        "caption": frame.caption,
                        "object": frame.object,
                        "bbox": frame.bbox,
                        "intensity": frame.intensity,
                    }
                    for frame in self.frames
                ],
            }
        )


_OBJECTS = [
    "rover",
    "antenna",
    "buoy",
    "diver",
    "beacon",
    "drone",
]


_CAPTION_VERBS = [
    "approaches",
    "observes",
    "stabilises",
    "records",
    "calibrates",
]


def build_videoqa_registry() -> Registry:
    registry = Registry()
    registry.add_field("question", keys=["clip_id"], salience=True, modality="text")
    registry.add_field(
        "frames", keys=["clip_id", "frame_id"], salience=True, modality="text"
    )
    registry.add_field("layout", keys=["clip_id", "frame_id"], modality="layout")
    registry.add_field("vision", keys=["clip_id", "frame_id"], modality="vision")
    registry.add_affinity("frames", "layout", keys=["clip_id", "frame_id"])
    registry.add_affinity("vision", "layout", keys=["clip_id", "frame_id"])
    registry.validate()
    return registry


def build_videoqa_encoders(
    registry: Registry,
    *,
    question_dim: int = 12,
    frame_dim: int = 12,
    layout_dim: int = 6,
    vision_dim: int = 6,
) -> Dict[str, Encoder]:
    encoders: Dict[str, Encoder] = {
        "question": TextEncoder(embedding_dim=question_dim),
        "frames": TextEncoder(embedding_dim=frame_dim),
        "layout": LayoutEncoder(embedding_dim=layout_dim),
        "vision": LayoutEncoder(embedding_dim=vision_dim),
    }
    for name, encoder in encoders.items():
        registry.register_encoder(name, encoder)
    return encoders


def synthetic_video_clip(
    clip_id: int,
    *,
    num_frames: int = 8,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed * 211 + clip_id * 31)
    target_frame = rng.randrange(max(num_frames, 1))
    target_object = rng.choice(_OBJECTS)
    frames: List[Dict[str, Any]] = []
    for frame_id in range(num_frames):
        obj = target_object if frame_id == target_frame else rng.choice(_OBJECTS)
        verb = rng.choice(_CAPTION_VERBS)
        caption = f"Frame {frame_id} {verb} the {obj}"
        intensity = 0.9 if frame_id == target_frame else 0.4 + 0.2 * rng.random()
        x1 = max(0.0, min(0.8, rng.random() * 0.8))
        y1 = max(0.0, min(0.8, rng.random() * 0.8))
        width = 0.15 + rng.random() * 0.1
        height = 0.15 + rng.random() * 0.1
        bbox = [x1, y1, min(1.0, x1 + width), min(1.0, y1 + height)]
        frames.append(
            {
                "clip_id": clip_id,
                "frame_id": frame_id,
                "caption": caption,
                "object": obj,
                "bbox": bbox,
                "intensity": intensity,
            }
        )
    question = f"Which frame shows the {target_object}?"
    return {
        "clip_id": clip_id,
        "question": question,
        "target_object": target_object,
        "target_frame": target_frame,
        "frames": frames,
    }


def synthetic_videoqa_dataset(
    count: int,
    *,
    seed: int = 0,
    num_frames: int = 8,
) -> List[Dict[str, Any]]:
    return [
        synthetic_video_clip(clip_id=index + 1, num_frames=num_frames, seed=seed)
        for index in range(count)
    ]


def videoqa_fields(
    document: Mapping[str, Any]
) -> Dict[str, List[MutableMapping[str, Any]]]:
    clip_id = int(document.get("clip_id", 0))
    frames = list(document.get("frames", []))
    fields: Dict[str, List[MutableMapping[str, Any]]] = {
        "question": [
            {
                "clip_id": clip_id,
                "text": str(document.get("question", "")),
                "target_object": document.get("target_object"),
            }
        ],
        "frames": [],
        "layout": [],
        "vision": [],
    }
    total_frames = float(max(len(frames) - 1, 1))
    for frame in frames:
        frame_id = int(frame.get("frame_id", 0))
        rel = frame_id / total_frames if total_frames else 0.0
        bbox = list(
            frame.get("bbox", [rel, rel, min(1.0, rel + 0.2), min(1.0, rel + 0.2)])
        )
        fields["frames"].append(
            {
                "clip_id": clip_id,
                "frame_id": frame_id,
                "text": str(frame.get("caption", "")),
                "object": frame.get("object"),
                "intensity": float(frame.get("intensity", 0.5)),
            }
        )
        fields["layout"].append(
            {
                "clip_id": clip_id,
                "frame_id": frame_id,
                "xyxy": bbox,
            }
        )
        fields["vision"].append(
            {
                "clip_id": clip_id,
                "frame_id": frame_id,
                "xyxy": bbox,
                "intensity": float(frame.get("intensity", 0.5)),
            }
        )
    return fields


def _videoqa_label(document: Mapping[str, Any]) -> str:
    return str(document.get("target_object", ""))


def _videoqa_predict(result: CompressionResult, document: Mapping[str, Any]) -> str:
    target_object = document.get("target_object")
    frames = result.compressed_fields.get("frames", [])
    for entry in frames:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("object")) == str(target_object):
            return str(entry.get("object"))
    return ""


def _videoqa_metadata(document: Mapping[str, Any], result: Any) -> Mapping[str, Any]:
    return {
        "clip_id": document.get("clip_id"),
        "target_frame": document.get("target_frame"),
        "kept_fields": {
            k: len(v) for k, v in getattr(result, "compressed_fields", {}).items()
        },
    }


def _videoqa_ablation_suite() -> Dict[str, AblationFn]:
    return {
        "drop_vision": _drop_field_ablation("vision"),
        "drop_layout": _drop_field_ablation("layout"),
        "shuffle_frames": _shuffle_field_ablation("frames"),
        "perturb_layout": _perturb_layout_ablation(scale=0.04),
        "noise_intensity": _noise_field_ablation("vision", "intensity", scale=0.3),
    }


def run_video_qa_benchmark(
    budget_values: Sequence[int] = (6, 10, 14),
    *,
    dataset_size: int = 10,
    seed: int = 0,
    num_frames: int = 8,
) -> Dict[str, Any]:
    registry = build_videoqa_registry()
    encoders = build_videoqa_encoders(registry)
    dataset = synthetic_videoqa_dataset(dataset_size, seed=seed, num_frames=num_frames)
    ablations = _videoqa_ablation_suite()

    runs: List[BudgetRun] = []
    for budget in budget_values:
        run = _evaluate_budget(
            budget=int(budget),
            dataset=dataset,
            registry_encoders=registry.encoders,
            fields_fn=videoqa_fields,
            label_fn=_videoqa_label,
            predict_fn=_videoqa_predict,
            metadata_fn=_videoqa_metadata,
            policy_name="video-qa-benchmark",
            retention_probe_sample_size=4,
            seed=seed,
            ablations=ablations,
        )
        runs.append(run)

    return {
        "dataset": "synthetic-video-qa",
        "dataset_size": len(dataset),
        "num_frames": num_frames,
        "budgets": [run.to_dict() for run in runs],
    }


__all__ = [
    "VideoClip",
    "VideoFrame",
    "build_videoqa_registry",
    "build_videoqa_encoders",
    "synthetic_video_clip",
    "synthetic_videoqa_dataset",
    "videoqa_fields",
    "run_video_qa_benchmark",
]
