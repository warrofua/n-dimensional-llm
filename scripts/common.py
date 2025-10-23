"""Shared helpers for ND-LLM training and evaluation scripts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from benchmarks.synthetic import (
    build_invoice_encoders,
    build_invoice_registry,
    high_value_label,
    invoice_fields,
    synthetic_invoice_dataset,
)
from nd_llm.bottleneck import IBottleneck
from nd_llm.model import NDEncoderDecoder


class InvoiceDataset(Dataset):
    """Thin dataset wrapper around the synthetic invoice benchmark."""

    def __init__(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        threshold: float,
    ) -> None:
        self._docs = list(documents)
        self._threshold = float(threshold)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._docs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        document = self._docs[index]
        fields = invoice_fields(document)
        doc_id = document.get("doc_id", index)
        sample_fields: Dict[str, List[Dict[str, Any]]] = {}
        for name, entries in fields.items():
            cloned: List[Dict[str, Any]] = []
            for entry in entries:
                cloned_entry = dict(entry)
                cloned_entry.setdefault("doc_id", doc_id)
                cloned.append(cloned_entry)
            sample_fields[name] = cloned
        label = 1 if high_value_label(document, self._threshold) else 0
        return {
            "fields": sample_fields,
            "target": int(label),
            "doc_id": doc_id,
        }


def build_invoice_model(
    *,
    hidden_dim: int = 128,
    num_classes: int = 2,
    scorer: Optional[Any] = None,
) -> NDEncoderDecoder:
    """Construct an :class:`NDEncoderDecoder` initialised for invoices."""

    registry = build_invoice_registry()
    encoders = build_invoice_encoders(registry)
    if scorer is not None:
        bottleneck = IBottleneck(target_budget=1, scorer_config=scorer)
    else:
        bottleneck = IBottleneck(target_budget=1)
    model = NDEncoderDecoder(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        bottleneck=bottleneck,
    )
    for name, spec in registry.fields.items():
        value_key: Optional[str]
        if name == "text":
            value_key = "text"
        else:
            value_key = None
        model.register_field(
            name,
            encoder=encoders[name],
            keys=spec.keys,
            salience=spec.salience,
            metadata=spec.metadata,
            value_key=value_key,
        )
    return model


def invoice_collate_fn(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge dataset samples into a batch compatible with the model."""

    combined_fields: Dict[str, List[Dict[str, Any]]] = {}
    targets: List[int] = []
    doc_ids: List[Any] = []
    for sample in samples:
        doc_id = sample.get("doc_id")
        doc_ids.append(doc_id)
        targets.append(int(sample.get("target", 0)))
        fields = sample.get("fields", {})
        for name, entries in fields.items():
            bucket = combined_fields.setdefault(name, [])
            for entry in entries:
                bucket.append(dict(entry))
    batch = {
        "fields": combined_fields,
        "targets": torch.tensor(targets, dtype=torch.long),
        "doc_ids": doc_ids,
    }
    return batch


def build_invoice_dataloader(
    *,
    dataset_size: int,
    threshold: float,
    batch_size: int = 2,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Return a :class:`~torch.utils.data.DataLoader` over synthetic invoices."""

    documents = synthetic_invoice_dataset(dataset_size, seed=seed)
    dataset = InvoiceDataset(documents, threshold=threshold)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=invoice_collate_fn,
    )


def average_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute simple accuracy ignoring ``-100`` sentinel targets."""

    if logits.numel() == 0 or targets.numel() == 0:
        return 0.0
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        valid = targets != -100
        if valid.sum() == 0:
            return 0.0
        correct = (predictions[valid] == targets[valid]).float().mean()
        return float(correct.detach().cpu())


__all__ = [
    "InvoiceDataset",
    "build_invoice_model",
    "build_invoice_dataloader",
    "invoice_collate_fn",
    "average_accuracy",
]
