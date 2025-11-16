"""Benchmark harnesses and synthetic datasets for ND-LLM."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "run_benchmark",
    "run_cord_benchmark",
    "run_chartqa_benchmark",
    "run_long_qa_benchmark",
    "run_video_qa_benchmark",
    "build_cord_registry",
    "build_cord_encoders",
    "cord_fields",
    "cord_high_total_label",
    "cord_total_amount",
    "load_cord_dataset",
    "build_chartqa_registry",
    "build_chartqa_encoders",
    "chartqa_fields",
    "chartqa_answer",
    "load_chartqa_dataset",
    "AmountEncoder",
    "build_invoice_encoders",
    "build_invoice_registry",
    "invoice_fields",
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
    if name == "run_cord_benchmark":
        return import_module("benchmarks.doc_understanding").run_cord_benchmark
    if name == "run_chartqa_benchmark":
        return import_module("benchmarks.chartqa").run_chartqa_benchmark
    if name == "run_long_qa_benchmark":
        return import_module("benchmarks.long_qa").run_long_qa_benchmark
    if name == "run_video_qa_benchmark":
        return import_module("benchmarks.video_qa").run_video_qa_benchmark
    if name in {
        "build_cord_registry",
        "build_cord_encoders",
        "cord_fields",
        "cord_high_total_label",
        "cord_total_amount",
        "load_cord_dataset",
    }:
        module = import_module("benchmarks.cord")
        return getattr(module, name)
    if name in {
        "build_chartqa_registry",
        "build_chartqa_encoders",
        "chartqa_fields",
        "chartqa_answer",
        "load_chartqa_dataset",
    }:
        module = import_module("benchmarks.chartqa")
        return getattr(module, name)
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
