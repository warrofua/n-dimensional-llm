"""Pipeline orchestration utilities."""

from __future__ import annotations

from typing import Any, Callable, Iterable


class Orchestrator:
    """Coordinate execution of encoder, bottleneck, and STM modules."""

    def __init__(self, steps: Iterable[Callable[..., Any]] | None = None) -> None:
        self._steps = list(steps or [])

    def add_step(self, step: Callable[..., Any]) -> None:
        """Add a callable execution step to the orchestration pipeline."""

        self._steps.append(step)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the configured pipeline with the provided arguments."""

        result: Any | None = None
        for step in self._steps:
            if result is None:
                result = step(*args, **kwargs)
            else:
                result = step(result)
        return result
