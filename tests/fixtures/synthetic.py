"""Synthetic dataset helpers exposed as pytest fixtures."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from benchmarks.synthetic import (
    AmountEncoder,
    build_invoice_encoders,
    build_invoice_registry,
    invoice_fields,
    synthetic_invoice,
    synthetic_invoice_dataset,
)


@pytest.fixture()
def invoice_registry():
    return build_invoice_registry()


@pytest.fixture()
def invoice_encoders(invoice_registry):  # type: ignore[override]
    return build_invoice_encoders(invoice_registry)


@pytest.fixture()
def invoice_sample() -> Dict[str, Any]:
    return synthetic_invoice(doc_id=1, seed=123)


@pytest.fixture()
def invoice_dataset() -> List[Dict[str, Any]]:
    return synthetic_invoice_dataset(5, seed=321)


__all__ = [
    "AmountEncoder",
    "invoice_dataset",
    "invoice_encoders",
    "invoice_fields",
    "invoice_registry",
    "invoice_sample",
    "synthetic_invoice",
    "synthetic_invoice_dataset",
]
