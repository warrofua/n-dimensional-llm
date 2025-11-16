"""Synthetic multimodal invoice dataset helpers for benchmarks and examples."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from nd_llm.encoders import Encoder, LayoutEncoder, TextEncoder
from nd_llm.registry import Registry


class AmountEncoder:
    """Simple encoder projecting monetary amounts to a scaled magnitude vector."""

    def __init__(self, embedding_dim: int = 2, scale: float = 0.05) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self._scale = float(scale)

    def encode(self, field_batch: Sequence[Any]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for item in field_batch:
            amount = _extract_amount(item)
            magnitude = amount * self._scale
            vector = [magnitude] + [0.0] * (self.embedding_dim - 1)
            embeddings.append(vector)
        return embeddings


def build_invoice_registry() -> Registry:
    """Create a registry describing the synthetic invoice schema."""

    registry = Registry()
    registry.add_field("text", keys=["doc_id", "line_id"], salience=True)
    registry.add_field("layout", keys=["doc_id", "line_id"])
    registry.add_field("amount", keys=["doc_id", "line_id"])
    registry.add_affinity("text", "layout", keys=["doc_id", "line_id"])
    registry.add_affinity("text", "amount", keys=["doc_id", "line_id"])
    registry.validate()
    return registry


def build_invoice_encoders(
    registry: Registry,
    *,
    text_dim: int = 8,
    layout_dim: int = 6,
    amount_dim: int = 2,
    amount_scale: float = 0.05,
) -> Dict[str, Encoder]:
    """Register encoder stubs for the invoice fields and return the mapping."""

    encoders: Dict[str, Encoder] = {
        "text": TextEncoder(embedding_dim=text_dim),
        "layout": LayoutEncoder(embedding_dim=layout_dim),
        "amount": AmountEncoder(embedding_dim=amount_dim, scale=amount_scale),
    }
    for field, encoder in encoders.items():
        registry.register_encoder(field, encoder)
    return encoders


@dataclass(frozen=True)
class SyntheticInvoiceLine:
    doc_id: int
    line_id: int
    description: str
    amount: float
    bbox: Mapping[str, Sequence[float]]
    salience: float


@dataclass(frozen=True)
class SyntheticInvoice:
    doc_id: int
    vendor: str
    lines: Sequence[SyntheticInvoiceLine]

    def as_fields(self) -> Dict[str, List[MutableMapping[str, Any]]]:
        return invoice_fields(
            {
                "doc_id": self.doc_id,
                "vendor": self.vendor,
                "lines": [
                    {
                        "doc_id": line.doc_id,
                        "line_id": line.line_id,
                        "description": line.description,
                        "amount": line.amount,
                        "bbox": line.bbox,
                        "salience": line.salience,
                    }
                    for line in self.lines
                ],
            }
        )


_VENDOR_NAMES = [
    "Acme Supplies",
    "Globex Services",
    "Initech Corp",
    "Soylent Foods",
    "Stark Manufacturing",
]

_DESCRIPTION_TOKENS = [
    "Consulting",
    "Hardware",
    "Software",
    "Support",
    "Maintenance",
    "Hosting",
    "Licensing",
]


def synthetic_invoice(
    doc_id: int,
    *,
    num_lines: int = 6,
    seed: int = 0,
    amount_range: tuple[float, float] = (80.0, 900.0),
) -> Dict[str, Any]:
    """Generate a deterministic synthetic invoice document."""

    rng = random.Random(seed + doc_id * 37)
    vendor = rng.choice(_VENDOR_NAMES)
    min_amount, max_amount = amount_range
    lines: List[Dict[str, Any]] = []
    for line_id in range(num_lines):
        amount = round(rng.uniform(min_amount, max_amount), 2)
        description = f"{rng.choice(_DESCRIPTION_TOKENS)} item {line_id + 1}"
        top = 0.15 + 0.08 * line_id
        bbox = {"xyxy": [0.1, top, 0.9, top + 0.06]}
        salience = min(1.0, 0.4 + (amount / max_amount))
        lines.append(
            {
                "doc_id": doc_id,
                "line_id": line_id,
                "description": description,
                "amount": amount,
                "bbox": bbox,
                "salience": salience,
            }
        )
    return {"doc_id": doc_id, "vendor": vendor, "lines": lines}


def synthetic_invoice_dataset(
    count: int,
    *,
    seed: int = 0,
    num_lines: int = 6,
    amount_range: tuple[float, float] = (80.0, 900.0),
) -> List[Dict[str, Any]]:
    """Return a list of synthetic invoices for benchmarking."""

    return [
        synthetic_invoice(
            doc_id=index + 1,
            num_lines=num_lines,
            seed=seed,
            amount_range=amount_range,
        )
        for index in range(count)
    ]


def invoice_fields(
    invoice: Mapping[str, Any]
) -> Dict[str, List[MutableMapping[str, Any]]]:
    """Convert an invoice dictionary to registry-aligned field batches."""

    doc_id = int(invoice.get("doc_id", 0))
    fields: Dict[str, List[MutableMapping[str, Any]]] = {
        "text": [],
        "layout": [],
        "amount": [],
    }
    for line in invoice.get("lines", []):
        line_id = int(line.get("line_id", 0))
        amount = float(line.get("amount", 0.0))
        bbox = line.get("bbox", {"xyxy": [0.0, 0.0, 1.0, 0.0]})
        description = str(line.get("description", ""))
        salience = float(line.get("salience", 0.5))
        fields["text"].append(
            {
                "doc_id": doc_id,
                "line_id": line_id,
                "text": description,
                "salience": salience,
            }
        )
        fields["layout"].append(
            {
                "doc_id": doc_id,
                "line_id": line_id,
                "xyxy": list(bbox.get("xyxy", [0.0, 0.0, 1.0, 1.0])),
            }
        )
        fields["amount"].append(
            {
                "doc_id": doc_id,
                "line_id": line_id,
                "amount": amount,
            }
        )
    return fields


def high_value_label(invoice: Mapping[str, Any], threshold: float) -> bool:
    """Return True if any line amount meets or exceeds ``threshold``."""

    return any(
        float(line.get("amount", 0.0)) >= threshold for line in invoice.get("lines", [])
    )


def _extract_amount(item: Any) -> float:
    if isinstance(item, Mapping):
        if "amount" in item:
            return float(item["amount"])
        if "value" in item:
            return float(item["value"])
    try:
        return float(item)
    except Exception as exc:  # pragma: no cover - defensive path
        raise TypeError(f"Cannot extract amount from item {item!r}") from exc


__all__ = [
    "AmountEncoder",
    "SyntheticInvoice",
    "SyntheticInvoiceLine",
    "build_invoice_encoders",
    "build_invoice_registry",
    "high_value_label",
    "invoice_fields",
    "synthetic_invoice",
    "synthetic_invoice_dataset",
]
