"""Typed helpers for working with the DocLayNet document-layout dataset."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from nd_llm.encoders import Encoder, LayoutEncoder, TextEncoder
from nd_llm.registry import Registry

__all__ = [
    "load_doclaynet_dataset",
    "build_doclaynet_registry",
    "build_doclaynet_encoders",
    "doclaynet_fields",
    "doclaynet_contains_table",
]


# A tiny in-memory sample used when the real dataset is unavailable.  The sample
# mirrors the minimal structure the conversion helpers expect, keeping runtime
# behaviour deterministic for doctests and local experiments.
_SAMPLE_DOCUMENT: Dict[str, Any] = {
    "doc_id": "doclaynet-sample",
    "page_id": "doclaynet-sample-0",
    "width": 1000,
    "height": 1400,
    "segments": [
        {
            "segment_id": 0,
            "label": "header",
            "confidence": 0.92,
            "polygon": [10, 10, 400, 10, 400, 120, 10, 120],
            "text": "DocLayNet Sample Heading",
            "tokens": [
                {
                    "token_id": 0,
                    "text": "DocLayNet",
                    "polygon": [10, 10, 180, 10, 180, 120, 10, 120],
                    "confidence": 0.9,
                },
                {
                    "token_id": 1,
                    "text": "Sample",
                    "polygon": [180, 10, 300, 10, 300, 120, 180, 120],
                    "confidence": 0.91,
                },
                {
                    "token_id": 2,
                    "text": "Heading",
                    "polygon": [300, 10, 400, 10, 400, 120, 300, 120],
                    "confidence": 0.93,
                },
            ],
        },
        {
            "segment_id": 1,
            "label": "paragraph",
            "confidence": 0.88,
            "polygon": [30, 180, 960, 180, 960, 620, 30, 620],
            "text": "This is a lightweight fallback paragraph used when the DocLayNet dataset is not installed.",
            "tokens": [
                {
                    "token_id": 0,
                    "text": "This",
                    "polygon": [30, 180, 110, 180, 110, 260, 30, 260],
                    "confidence": 0.87,
                },
                {
                    "token_id": 1,
                    "text": "is",
                    "polygon": [120, 180, 150, 180, 150, 260, 120, 260],
                    "confidence": 0.88,
                },
                {
                    "token_id": 2,
                    "text": "a",
                    "polygon": [160, 180, 190, 180, 190, 260, 160, 260],
                    "confidence": 0.9,
                },
                {
                    "token_id": 3,
                    "text": "lightweight",
                    "polygon": [200, 180, 370, 180, 370, 260, 200, 260],
                    "confidence": 0.9,
                },
                {
                    "token_id": 4,
                    "text": "fallback",
                    "polygon": [380, 180, 520, 180, 520, 260, 380, 260],
                    "confidence": 0.89,
                },
                {
                    "token_id": 5,
                    "text": "paragraph",
                    "polygon": [530, 180, 700, 180, 700, 260, 530, 260],
                    "confidence": 0.9,
                },
            ],
        },
    ],
    "metadata": {
        "source": "synthetic",
        "split": "sample",
    },
}


def load_doclaynet_dataset(
    root: Optional[Path | str] = None,
    *,
    split: str = "train",
    limit: Optional[int] = None,
    use_sample: bool = True,
) -> List[Dict[str, Any]]:
    """Load DocLayNet pages from ``root`` or return a bundled sample."""

    if root is None:
        if not use_sample:
            raise FileNotFoundError(
                "DocLayNet root path is required when use_sample is False"
            )
        return list(_load_sample(limit))

    path = Path(root)
    if not path.exists():
        if use_sample:
            return list(_load_sample(limit))
        raise FileNotFoundError(f"DocLayNet root directory '{path}' does not exist")

    pages = list(_load_from_directory(path, split=split, limit=limit))
    if pages:
        return pages

    if use_sample:
        return list(_load_sample(limit))

    raise FileNotFoundError(
        f"DocLayNet split '{split}' not found in '{path}' and no bundled sample is available"
    )


def build_doclaynet_registry() -> Registry:
    """Return a registry that captures DocLayNet text, layout, and segment metadata."""

    registry = Registry()
    registry.add_field(
        "text",
        keys=["doc_id", "page_id", "segment_id", "token_id"],
        salience=True,
        modality="text",
    )
    registry.add_field(
        "layout",
        keys=["doc_id", "page_id", "segment_id", "token_id"],
        modality="layout",
    )
    registry.add_field(
        "segment", keys=["doc_id", "page_id", "segment_id"], modality="entity"
    )
    registry.add_affinity("segment", "text", keys=["doc_id", "page_id", "segment_id"])
    registry.add_affinity("text", "layout", keys=["doc_id", "page_id", "token_id"])
    registry.validate()
    return registry


def build_doclaynet_encoders(
    registry: Registry,
    *,
    text_dim: int = 16,
    layout_dim: int = 8,
    segment_dim: int = 4,
) -> Dict[str, Encoder]:
    """Register simple encoder stubs for the DocLayNet registry."""

    encoders: Dict[str, Encoder] = {
        "text": TextEncoder(embedding_dim=text_dim),
        "layout": LayoutEncoder(embedding_dim=layout_dim),
        "segment": TextEncoder(embedding_dim=segment_dim),
    }
    for field, encoder in encoders.items():
        registry.register_encoder(field, encoder)
    return encoders


def doclaynet_fields(
    document: Mapping[str, Any],
) -> Dict[str, List[MutableMapping[str, Any]]]:
    """Convert a DocLayNet page into registry-aligned field payloads."""

    doc_id = str(document.get("doc_id") or document.get("document_id") or "")
    page_id = str(document.get("page_id") or document.get("page") or 0)
    width = float(document.get("width") or document.get("img_width") or 1.0)
    height = float(document.get("height") or document.get("img_height") or 1.0)
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    text_field: List[MutableMapping[str, Any]] = []
    layout_field: List[MutableMapping[str, Any]] = []
    segment_field: List[MutableMapping[str, Any]] = []

    raw_segments = document.get("segments")
    if not isinstance(raw_segments, Sequence):
        raw_segments = []

    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            continue
        segment_id = int(
            raw_segment.get("segment_id") or raw_segment.get("id") or index
        )
        label = str(
            raw_segment.get("label") or raw_segment.get("category") or "segment"
        )
        confidence_value = raw_segment.get("confidence")
        confidence = float(confidence_value) if confidence_value is not None else 0.0
        segment_polygon = _coerce_polygon(
            raw_segment.get("polygon") or raw_segment.get("bbox")
        )

        tokens = raw_segment.get("tokens") or raw_segment.get("words")
        prepared_tokens = _prepare_tokens(
            tokens, fallback_text=str(raw_segment.get("text", ""))
        )

        token_ids = [int(token["token_id"]) for token in prepared_tokens]

        segment_field.append(
            {
                "doc_id": doc_id,
                "page_id": page_id,
                "segment_id": segment_id,
                "label": label,
                "confidence": confidence,
                "polygon": segment_polygon,
                "token_ids": token_ids,
            }
        )

        for token in prepared_tokens:
            token_id = int(token["token_id"])
            token_text = str(token["text"])
            token_confidence = float(token["confidence"])
            token_polygon = _coerce_polygon(token.get("polygon"))
            bbox = _polygon_to_bbox(token_polygon)
            norm_bbox = _normalise_box(bbox, width, height)

            text_field.append(
                {
                    "doc_id": doc_id,
                    "page_id": page_id,
                    "token_id": token_id,
                    "text": token_text,
                    "segment_id": segment_id,
                    "segment_label": label,
                    "confidence": token_confidence,
                }
            )
            layout_field.append(
                {
                    "doc_id": doc_id,
                    "page_id": page_id,
                    "token_id": token_id,
                    "xyxy": norm_bbox,
                    "polygon": token_polygon,
                    "segment_id": segment_id,
                }
            )

    return {"text": text_field, "layout": layout_field, "segment": segment_field}


def doclaynet_contains_table(document: Mapping[str, Any]) -> bool:
    """Return ``True`` when a DocLayNet page includes a table segment."""

    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("contains_table", "containsTable", "has_table"):
            flag = metadata.get(key)
            coerced = _coerce_bool(flag)
            if coerced is not None:
                return coerced

    segments = document.get("segments") or document.get("entities")
    if isinstance(segments, Sequence):
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            label = (
                segment.get("label") or segment.get("category") or segment.get("type")
            )
            if isinstance(label, str) and "table" in label.lower():
                return True

            segment_metadata = segment.get("metadata")
            if isinstance(segment_metadata, Mapping):
                flag = segment_metadata.get("contains_table")
                coerced = _coerce_bool(flag)
                if coerced:
                    return True

    return False


def _prepare_tokens(
    tokens: Optional[Iterable[Any]],
    *,
    fallback_text: str,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    if tokens is None:
        tokens = []

    for index, raw_token in enumerate(tokens):
        if not isinstance(raw_token, Mapping):
            continue
        token_id = int(raw_token.get("token_id") or raw_token.get("id") or index)
        text = str(raw_token.get("text") or "")
        if not text and fallback_text:
            text = fallback_text
        confidence_value = raw_token.get("confidence")
        confidence = float(confidence_value) if confidence_value is not None else 0.0
        polygon = _coerce_polygon(raw_token.get("polygon") or raw_token.get("bbox"))
        prepared.append(
            {
                "token_id": token_id,
                "text": text,
                "confidence": confidence,
                "polygon": polygon,
            }
        )

    if not prepared and fallback_text:
        prepared.append(
            {
                "token_id": 0,
                "text": fallback_text,
                "confidence": 0.0,
                "polygon": [],
            }
        )

    return prepared


def _load_sample(limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    count = 0
    while True:
        yield deepcopy(_SAMPLE_DOCUMENT)
        count += 1
        if limit is not None and count >= limit:
            break
        if limit is None:
            break


def _load_from_directory(
    root: Path, *, split: str, limit: Optional[int]
) -> Iterator[Dict[str, Any]]:
    split_path = root / split
    if not split_path.exists():
        return

    count = 0
    for path in sorted(split_path.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        document = _prepare_document(data, default_doc_id=path.stem)
        yield document
        count += 1
        if limit is not None and count >= limit:
            return


def _prepare_document(raw: Mapping[str, Any], *, default_doc_id: str) -> Dict[str, Any]:
    doc_id = str(raw.get("doc_id") or raw.get("id") or default_doc_id)
    page_id = str(raw.get("page_id") or raw.get("page") or 0)
    width_value = raw.get("width") or raw.get("img_width") or raw.get("page_width")
    height_value = raw.get("height") or raw.get("img_height") or raw.get("page_height")
    width = int(width_value) if width_value is not None else 0
    height = int(height_value) if height_value is not None else 0
    metadata_raw = raw.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
    metadata.setdefault("doc_id", doc_id)
    metadata.setdefault("page_id", page_id)
    metadata.setdefault("split", str(raw.get("split") or metadata.get("split") or ""))

    raw_segments = raw.get("segments") or raw.get("entities") or []
    segments: List[Dict[str, Any]] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            continue
        segment_id = int(
            raw_segment.get("segment_id") or raw_segment.get("id") or index
        )
        label = str(
            raw_segment.get("label") or raw_segment.get("category") or "segment"
        )
        confidence_value = raw_segment.get("confidence")
        confidence = float(confidence_value) if confidence_value is not None else 0.0
        polygon = _coerce_polygon(raw_segment.get("polygon") or raw_segment.get("bbox"))
        tokens = _prepare_tokens(
            raw_segment.get("tokens") or raw_segment.get("words"),
            fallback_text=str(raw_segment.get("text", "")),
        )
        segments.append(
            {
                "segment_id": segment_id,
                "label": label,
                "confidence": confidence,
                "polygon": polygon,
                "tokens": tokens,
            }
        )

    document: Dict[str, Any] = {
        "doc_id": doc_id,
        "page_id": page_id,
        "width": width,
        "height": height,
        "segments": segments,
        "metadata": metadata,
    }
    return document


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _coerce_polygon(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        coords = []
        for key in ("x1", "y1", "x2", "y2"):
            coord_value = value.get(key)
            if coord_value is None:
                continue
            coords.append(float(coord_value))
        if coords:
            return coords
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        coords = [float(v) for v in value if isinstance(v, (int, float))]
        return coords
    return []


def _polygon_to_bbox(polygon: Sequence[float]) -> Tuple[float, float, float, float]:
    if not polygon:
        return (0.0, 0.0, 0.0, 0.0)
    xs = polygon[0::2]
    ys = polygon[1::2]
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return (float(x1), float(y1), float(x2), float(y2))


def _normalise_box(
    box: Tuple[float, float, float, float], width: float, height: float
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0
    return (x1 / width, y1 / height, x2 / width, y2 / height)
