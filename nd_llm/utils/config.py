"""Configuration dataclasses for STM storage and orchestration policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class STMConfig:
    """Configuration for the short-term memory (STM) storage backend.

    Attributes
    ----------
    storage_dir:
        Directory where tensor payloads and index metadata are persisted.
    index_filename:
        Name of the JSON index file relative to ``storage_dir``.
    """

    storage_dir: Union[str, Path]
    index_filename: str = "index.json"

    def __post_init__(self) -> None:  # type: ignore[override]
        storage_path = Path(self.storage_dir)
        object.__setattr__(self, "storage_dir", storage_path)
        if not self.index_filename:
            raise ValueError("index_filename must be a non-empty string")


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for orchestration policies and budget handling."""

    target_budget: float
    policy_name: str = "default"
    budget_step: float = 0.1
    retention_probe_sample_size: int = 10

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.target_budget < 0:
            raise ValueError("target_budget must be non-negative")
        if self.budget_step <= 0:
            raise ValueError("budget_step must be positive")
        if self.retention_probe_sample_size <= 0:
            raise ValueError("retention_probe_sample_size must be positive")
