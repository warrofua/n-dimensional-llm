"""Field adapters and canonical alignment utilities for ND-LLM inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

Document = Mapping[str, Any]
Entry = MutableMapping[str, Any]
BuilderFn = Callable[[Document], Iterable[Entry]]
AlignerFn = Callable[[Document, Entry], Optional[Sequence[float]]]


def _ensure_sequence(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return [_ensure_number(value.get(key, 0.0)) for key in ("x1", "y1", "x2", "y2")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_ensure_number(val) for val in value[:4]]
    return None


def _ensure_number(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # pragma: no cover - defensive guardrail
        return 0.0


def _normalise_quad(quad: Any) -> Dict[str, float]:
    if isinstance(quad, Mapping):
        keys = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
        return {key: _ensure_number(quad.get(key, 0.0)) for key in keys}
    if isinstance(quad, Sequence) and len(quad) >= 8:
        return {
            "x1": _ensure_number(quad[0]),
            "y1": _ensure_number(quad[1]),
            "x2": _ensure_number(quad[2]),
            "y2": _ensure_number(quad[3]),
            "x3": _ensure_number(quad[4]),
            "y3": _ensure_number(quad[5]),
            "x4": _ensure_number(quad[6]),
            "y4": _ensure_number(quad[7]),
        }
    return {key: 0.0 for key in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")}


def quad_to_box(quad: Any) -> List[float]:
    normalised = _normalise_quad(quad)
    xs = [normalised["x1"], normalised["x2"], normalised["x3"], normalised["x4"]]
    ys = [normalised["y1"], normalised["y2"], normalised["y3"], normalised["y4"]]
    return [
        min(xs) if xs else 0.0,
        min(ys) if ys else 0.0,
        max(xs) if xs else 0.0,
        max(ys) if ys else 0.0,
    ]


def normalise_box(box: Sequence[float], *, width: float, height: float) -> List[float]:
    """Normalise absolute coordinates to the unit square."""

    safe_width = max(float(width) or 1.0, 1.0)
    safe_height = max(float(height) or 1.0, 1.0)
    if not box:
        x1 = y1 = x2 = y2 = 0.0
    else:
        x1 = _ensure_number(box[0])
        y1 = _ensure_number(box[1] if len(box) > 1 else 0.0)
        x2 = _ensure_number(box[2] if len(box) > 2 else x1)
        y2 = _ensure_number(box[3] if len(box) > 3 else y1)
    return [
        max(0.0, min(1.0, x1 / safe_width)),
        max(0.0, min(1.0, y1 / safe_height)),
        max(0.0, min(1.0, x2 / safe_width)),
        max(0.0, min(1.0, y2 / safe_height)),
    ]


class LayoutAligner:
    """Align layout-aware entries into document-normalised coordinates."""

    def __init__(
        self,
        *,
        quad_key: str = "quad",
        width_key: str = "width",
        height_key: str = "height",
    ) -> None:
        self._quad_key = quad_key
        self._width_key = width_key
        self._height_key = height_key

    def __call__(self, document: Document, entry: Entry) -> Optional[List[float]]:
        quad = entry.get(self._quad_key) or entry.get("coords") or entry.get("xyxy")
        if quad is None:
            return None
        width = document.get(self._width_key, 0.0)
        height = document.get(self._height_key, 0.0)
        box = quad_to_box(quad)
        return normalise_box(
            box, width=float(width or 1.0), height=float(height or 1.0)
        )


@dataclass
class FieldAdapter:
    """Declarative adapter that prepares per-field entries with canonical coords."""

    name: str
    builder: BuilderFn
    aligner: Optional[AlignerFn] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def adapt(self, document: Document) -> List[Entry]:
        adapted: List[Entry] = []
        for raw in self.builder(document):
            entry: Entry = dict(raw)
            if self.aligner is not None:
                coords = self.aligner(document, entry)
                if coords is not None:
                    entry["coords"] = list(coords)
                    # Provide xyxy alias to satisfy existing consumers.
                    entry.setdefault("xyxy", list(coords))
            adapted.append(entry)
        return adapted


class FieldAdapterRegistry:
    """Container that applies registered adapters to incoming documents."""

    def __init__(self) -> None:
        self._adapters: Dict[str, FieldAdapter] = {}

    def register(self, adapter: FieldAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"Field adapter '{adapter.name}' already registered")
        self._adapters[adapter.name] = adapter

    def transform(self, document: Document) -> Dict[str, List[Entry]]:
        return {
            name: adapter.adapt(document) for name, adapter in self._adapters.items()
        }

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)

    def __iter__(self):
        return iter(self._adapters.items())
