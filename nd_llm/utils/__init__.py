"""Utility helpers and configuration objects for the n-dimensional LLM package."""

from .canonical_cells import (
    DEFAULT_BACKEND,
    NUMPY_AVAILABLE,
    TORCH_AVAILABLE,
    aggregate_fields,
    assign_to_cells,
    rasterize_cells,
)
from .config import OrchestratorConfig, STMConfig
from .fields import PackedFields, pack_fields
from .mi import build_mi_proxy_context

__all__ = [
    "DEFAULT_BACKEND",
    "NUMPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "aggregate_fields",
    "assign_to_cells",
    "OrchestratorConfig",
    "PackedFields",
    "STMConfig",
    "pack_fields",
    "rasterize_cells",
    "build_mi_proxy_context",
]
