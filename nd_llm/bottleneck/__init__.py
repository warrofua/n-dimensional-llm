"""Bottleneck implementations for ND-LLM."""

from .ib import (
    CompressionResult,
    CompressionTelemetry,
    IBottleneck,
    NormScoringStrategy,
    QueryDotProductScoringStrategy,
    RegistryAwareBudgetAllocator,
)
from .learnable import LearnableScoringStrategy, LearnableTokenScorer, configure_scorer

__all__ = [
    "IBottleneck",
    "CompressionResult",
    "CompressionTelemetry",
    "NormScoringStrategy",
    "QueryDotProductScoringStrategy",
    "RegistryAwareBudgetAllocator",
    "LearnableTokenScorer",
    "LearnableScoringStrategy",
    "configure_scorer",
]
