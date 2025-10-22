"""Encoder definitions for nd_llm."""

from __future__ import annotations

from typing import Any


def encode_identity(value: Any) -> Any:
    """Identity encoder stub that returns the value unchanged."""

    return value


__all__ = ["encode_identity"]
