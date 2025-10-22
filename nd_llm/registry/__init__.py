"""Lightweight registry utilities for ND-LLM prototypes."""
from __future__ import annotations

from typing import Dict, Mapping

from nd_llm.encoders import Encoder


class Registry:
    """Registry for associating field names with encoder instances."""

    def __init__(self) -> None:
        self._encoders: Dict[str, Encoder] = {}

    def register_encoder(self, field: str, encoder: Encoder) -> None:
        self._encoders[field] = encoder

    def get_encoder(self, field: str) -> Encoder:
        try:
            return self._encoders[field]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"encoder not registered for field '{field}'") from exc

    @property
    def encoders(self) -> Mapping[str, Encoder]:
        return dict(self._encoders)


__all__ = ["Registry"]
