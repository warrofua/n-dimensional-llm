"""Bottleneck implementations for ND-LLM."""
from .ib import (
    CompressionResult,
    CompressionTelemetry,
    IBottleneck,
    NormScoringStrategy,
    QueryDotProductScoringStrategy,
    RegistryAwareBudgetAllocator,
)

__all__ = [
    "IBottleneck",
    "CompressionResult",
    "CompressionTelemetry",
    "NormScoringStrategy",
    "QueryDotProductScoringStrategy",
    "RegistryAwareBudgetAllocator",
]
