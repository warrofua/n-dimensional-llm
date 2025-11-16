"""Public registry exports for ND-LLM."""

from .models import AffinityRule, FieldSpec, Registry
from .adapters import (
    FieldAdapter,
    FieldAdapterRegistry,
    LayoutAligner,
    normalise_box,
    quad_to_box,
)

__all__ = [
    "AffinityRule",
    "FieldSpec",
    "FieldAdapter",
    "FieldAdapterRegistry",
    "LayoutAligner",
    "normalise_box",
    "quad_to_box",
    "Registry",
]
