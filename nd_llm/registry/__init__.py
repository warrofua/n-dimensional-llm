"""Registry components for nd_llm."""

from __future__ import annotations

from typing import Any, Dict, ItemsView


class Registry:
    """Simple placeholder registry implementation.

    This stub can be replaced with a feature-complete registry that
    tracks encoders, bottlenecks, and other components. For now it only
    stores items in an in-memory dictionary.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def register(self, name: str, item: Any) -> None:
        """Register an item under the provided *name*."""

        self._items[name] = item

    def get(self, name: str) -> Any:
        """Retrieve a previously registered item."""

        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> ItemsView[str, Any]:
        """Return a dynamic view of the registered items."""

        return self._items.items()
