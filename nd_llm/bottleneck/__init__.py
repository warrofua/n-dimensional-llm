"""Information bottleneck interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IBottleneck(ABC):
    """Abstract base class for information bottleneck modules."""

    @abstractmethod
    def compress(self, *args: Any, **kwargs: Any) -> Any:
        """Reduce the dimensionality of the provided inputs."""

    @abstractmethod
    def decompress(self, *args: Any, **kwargs: Any) -> Any:
        """Reconstruct the original representation from the bottleneck."""
