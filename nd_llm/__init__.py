"""Top-level package for nd_llm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .bottleneck import IBottleneck
from .orchestration import Orchestrator
from .registry import Registry
from .stm import STM

try:
    __version__ = version("nd-llm")
except PackageNotFoundError:  # pragma: no cover - fallback when package not installed
    __version__ = "0.1.0"

__all__ = ["__version__", "Registry", "IBottleneck", "STM", "Orchestrator"]
