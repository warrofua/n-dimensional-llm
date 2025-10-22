"""ND-LLM core package."""

from .registry import Registry, FieldSpec, AffinityRule  # noqa: F401
"""Core package for n-dimensional LLM utilities."""

__all__ = ["stm", "orchestration", "utils"]
