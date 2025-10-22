"""Benchmark harnesses and synthetic datasets for ND-LLM."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "run_benchmark",
    "run_funsd_benchmark",
    "run_long_qa_benchmark",
    "run_video_qa_benchmark",
    "AmountEncoder",
    "build_invoice_encoders",
    "build_invoice_registry",
    "build_funsd_encoders",
    "build_funsd_registry",
    "funsd_fields",
    "funsd_numeric_answer_label",
    "invoice_fields",
    "load_funsd_dataset",
    "synthetic_invoice",
    "synthetic_invoice_dataset",
    "build_longqa_registry",
    "build_longqa_encoders",
    "synthetic_longqa_dialog",
    "synthetic_longqa_dataset",
    "longqa_fields",
    "build_videoqa_registry",
    "build_videoqa_encoders",
    "synthetic_video_clip",
    "synthetic_videoqa_dataset",
    "videoqa_fields",
    "LongQADialog",
    "LongQASegment",
    "VideoClip",
    "VideoFrame",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin convenience wrapper
    if name == "run_benchmark":
        return import_module("benchmarks.doc_understanding").run_benchmark
    if name == "run_funsd_benchmark":
        return import_module("benchmarks.doc_understanding").run_funsd_benchmark
    if name == "run_long_qa_benchmark":
        return import_module("benchmarks.long_qa").run_long_qa_benchmark
    if name == "run_video_qa_benchmark":
        return import_module("benchmarks.video_qa").run_video_qa_benchmark
    if name in {
        "AmountEncoder",
        "build_invoice_encoders",
        "build_invoice_registry",
        "invoice_fields",
        "synthetic_invoice",
        "synthetic_invoice_dataset",
    }:
        module = import_module("benchmarks.synthetic")
        return getattr(module, name)
    if name in {
        "build_funsd_encoders",
        "build_funsd_registry",
        "funsd_fields",
        "funsd_numeric_answer_label",
        "load_funsd_dataset",
    }:
        module = import_module("benchmarks.funsd")
        return getattr(module, name)
    if name in {
        "build_longqa_registry",
        "build_longqa_encoders",
        "synthetic_longqa_dialog",
        "synthetic_longqa_dataset",
        "longqa_fields",
        "LongQADialog",
        "LongQASegment",
    }:
        module = import_module("benchmarks.long_qa")
        return getattr(module, name)
    if name in {
        "build_videoqa_registry",
        "build_videoqa_encoders",
        "synthetic_video_clip",
        "synthetic_videoqa_dataset",
        "videoqa_fields",
        "VideoClip",
        "VideoFrame",
    }:
        module = import_module("benchmarks.video_qa")
        return getattr(module, name)
    raise AttributeError(f"module 'benchmarks' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial metadata
    return sorted(set(globals().get("__all__", [])))
