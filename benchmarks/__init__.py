"""Benchmark harnesses and synthetic datasets for ND-LLM."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "run_benchmark",
    "AmountEncoder",
    "build_invoice_encoders",
    "build_invoice_registry",
    "invoice_fields",
    "synthetic_invoice",
    "synthetic_invoice_dataset",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin convenience wrapper
    if name == "run_benchmark":
        return import_module("benchmarks.doc_understanding").run_benchmark
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
    raise AttributeError(f"module 'benchmarks' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial metadata
    return sorted(set(globals().get("__all__", [])))
