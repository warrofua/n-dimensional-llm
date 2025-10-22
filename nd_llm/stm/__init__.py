"""Short-term memory (STM) components."""

from __future__ import annotations

from typing import Any, List


class STM:
    """A lightweight short-term memory buffer stub."""

    def __init__(self, capacity: int = 10) -> None:
        self.capacity = capacity
        self._buffer: List[Any] = []

    def push(self, item: Any) -> None:
        """Add an item to the STM buffer, evicting the oldest entry if needed."""

        self._buffer.append(item)
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def snapshot(self) -> List[Any]:
        """Return a copy of the current buffer."""

        return list(self._buffer)
