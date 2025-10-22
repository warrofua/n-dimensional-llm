"""Encoder protocol and concrete stubs for ND-LLM."""
from __future__ import annotations

import hashlib
from itertools import cycle
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


FieldBatch = Sequence[Any]
EmbeddingBatch = Sequence[Sequence[float]]


@runtime_checkable
class Encoder(Protocol):
    """Minimal protocol for encoders used by the bottleneck."""

    embedding_dim: int

    def encode(self, field_batch: FieldBatch) -> EmbeddingBatch:
        """Encode a batch of field items into deterministic embeddings."""


def _hash_to_unit_float(value: bytes, dim: int) -> list[float]:
    """Map arbitrary bytes to a deterministic float vector in [0, 1]."""

    digest = hashlib.sha256(value).digest() or b"\x00"
    repeated = cycle(digest)
    return [next(repeated) / 255.0 for _ in range(dim)]


class TextEncoder:
    """Deterministic text encoder stub."""

    def __init__(self, embedding_dim: int = 8) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, field_batch: FieldBatch) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for item in field_batch:
            text = "" if item is None else str(item)
            embeddings.append(_hash_to_unit_float(text.encode("utf-8"), self.embedding_dim))
        return embeddings


class LayoutEncoder:
    """Layout encoder stub that projects coordinates into a dense embedding."""

    def __init__(self, embedding_dim: int = 6) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, field_batch: FieldBatch) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for item in field_batch:
            embeddings.append(self._encode_single(item))
        return embeddings

    def _encode_single(self, item: Any) -> list[float]:
        values: list[float] = []
        if isinstance(item, Mapping):
            if "xyxy" in item and isinstance(item["xyxy"], Sequence):
                values.extend(float(v) for v in item["xyxy"][:4])
            for key in ("x1", "y1", "x2", "y2", "width", "height", "x", "y"):
                if key in item:
                    values.append(float(item[key]))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            values.extend(float(v) for v in item[: self.embedding_dim])

        if not values:
            serialized = str(item).encode("utf-8") if item is not None else b""
            return _hash_to_unit_float(serialized, self.embedding_dim)

        if len(values) >= self.embedding_dim:
            return [float(v) for v in values[: self.embedding_dim]]

        repeated: list[float] = []
        idx = 0
        while len(repeated) < self.embedding_dim:
            repeated.append(float(values[idx % len(values)]))
            idx += 1
        return repeated
