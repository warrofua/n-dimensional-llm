"""Bottleneck implementations for ND-LLM."""
from .ib import CompressionResult, CompressionTelemetry, IBottleneck

__all__ = ["IBottleneck", "CompressionResult", "CompressionTelemetry"]
