"""Utility helpers for nd_llm."""

from __future__ import annotations

from typing import Any


def ensure_list(value: Any) -> list[Any]:
    """Return *value* as a list."""

    if isinstance(value, list):
        return value
    return [value]


__all__ = ["ensure_list"]
