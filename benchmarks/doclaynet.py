"""Utilities for loading and normalising the DocLayNet dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from nd_llm.encoders import Encoder, LayoutEncoder, TextEncoder
from nd_llm.registry import Registry

_SAMPLE_PATH = Path(__file__).with_name("data").joinpath("doclaynet_sample.jsonl")

_DOCLAYNET_SPLITS: Dict[str, List[str]] = {
    "train": ["train", "training"],
    "training": ["train", "training"],
    "test": ["test", "testing"],
    "validation": ["validation", "valid", "val", "dev"],
    "val": ["validation", "valid", "val", "dev"],
}


def load_doclaynet_dataset(
    root: Optional[Path | str] = None,
    *,
    split: str = "train",
    limit: Optional[int] = None,
    use_sample: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Load DocLayNet documents from ``root`` or fall back to the bundled sample."""

    documents: List[Dict[str, Any]] = []
    if use_sample or root is None:
        documents = list(_load_sample(limit))
    else:
        path = Path(root)
        if not path.exists():
            raise FileNotFoundError(f"DocLayNet root directory '{path}' does not exist")
        documents = list(_load_from_directory(path, split=split, limit=limit))

    if not documents and not use_sample and root is None:
        documents = list(_load_sample(limit))
    return documents


def build_doclaynet_registry() -> Registry:
    """Return a registry describing DocLayNet token, layout and region fields."""

    registry = Registry()
    registry.add_field("text", keys=["doc_id", "token_id"], salience=True, modality="text")
    registry.add_field("layout", keys=["doc_id", "token_id"], modality="layout")
    registry.add_field("region", keys=["doc_id", "region_id"], modality="entity")
    registry.add_affinity("text", "layout", keys=["doc_id", "token_id"])
    registry.add_affinity("region", "text", keys=["doc_id"])
    registry.add_affinity("region", "layout", keys=["doc_id"])
    registry.validate()
    return registry


def build_doclaynet_encoders(
    registry: Registry,
    *,
    text_dim: int = 8,
    layout_dim: int = 6,
    region_dim: int = 4,
) -> Dict[str, Encoder]:
    """Register simple encoder stubs for the DocLayNet fields."""

    encoders: Dict[str, Encoder] = {
        "text": TextEncoder(embedding_dim=text_dim),
        "layout": LayoutEncoder(embedding_dim=layout_dim),
        "region": TextEncoder(embedding_dim=region_dim),
    }
    for field, encoder in encoders.items():
        registry.register_encoder(field, encoder)
    return encoders


def doclaynet_fields(document: Mapping[str, Any]) -> Dict[str, List[MutableMapping[str, Any]]]:
    """Convert a DocLayNet document into registry-aligned field batches."""

    doc_id = str(document.get("doc_id") or document.get("id") or document.get("document_id") or "")
    width, height = _resolve_size(document)
    base_page = _resolve_page_index(document)
    tokens = _extract_tokens(document)
    regions, token_to_region = _collect_regions(document)

    text_field: List[MutableMapping[str, Any]] = []
    layout_field: List[MutableMapping[str, Any]] = []
    region_field: List[MutableMapping[str, Any]] = []
    token_text: Dict[int, str] = {}

    for token_index, raw_token in enumerate(tokens):
        token_id = _coerce_int(raw_token.get("id") or raw_token.get("token_id") or raw_token.get("idx"))
        if token_id is None:
            token_id = token_index
        token_id = int(token_id)

        page_index = _resolve_page_index(raw_token, default=base_page)
        region_id = _coerce_int(
            raw_token.get("region_id")
            or raw_token.get("regionId")
            or raw_token.get("region")
            or token_to_region.get(token_id)
        )
        if region_id is None:
            region_id = -1
        region_id = int(region_id)

        label = _resolve_region_label(regions.get(region_id))
        text_value = _resolve_token_text(raw_token)
        token_text[token_id] = text_value

        text_field.append(
            {
                "doc_id": doc_id,
                "token_id": token_id,
                "text": text_value,
                "region_id": region_id,
                "region_label": label,
                "page_index": page_index,
            }
        )

        bbox = raw_token.get("bbox") or raw_token.get("box") or raw_token.get("xyxy")
        layout_field.append(
            {
                "doc_id": doc_id,
                "token_id": token_id,
                "xyxy": _normalise_box(bbox, width, height),
                "region_id": region_id,
                "page_index": page_index,
            }
        )

    for region_id, region in regions.items():
        token_ids = list({int(token_id) for token_id in region.get("token_ids", [])})
        label = _resolve_region_label(region)
        text_value = region.get("text")
        if not text_value:
            text_value = " ".join(token_text.get(token_id, "") for token_id in token_ids).strip()
        page_index = _resolve_page_index(region, default=base_page)
        bbox = region.get("bbox") or region.get("box") or region.get("xyxy")
        region_field.append(
            {
                "doc_id": doc_id,
                "region_id": int(region_id),
                "label": label,
                "token_ids": token_ids,
                "xyxy": _normalise_box(bbox, width, height),
                "text": text_value,
                "page_index": page_index,
            }
        )

    return {"text": text_field, "layout": layout_field, "region": region_field}


def doclaynet_contains_table(document: Mapping[str, Any]) -> bool:
    """Return ``True`` if the document contains a region labelled as a table."""

    regions, _ = _collect_regions(document)
    for region in regions.values():
        label = _resolve_region_label(region)
        if label.lower() == "table":
            return True
    tokens = _extract_tokens(document)
    for token in tokens:
        text_value = _resolve_token_text(token)
        if "table" in text_value.lower():
            return True
    return False


def _load_sample(limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    count = 0
    path = _SAMPLE_PATH
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            document = json.loads(line)
            yield _prepare_document(document, document.get("id"))
            count += 1
            if limit is not None and count >= limit:
                break


def _load_from_directory(root: Path, *, split: str, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    split_key = split.lower()
    candidates = _DOCLAYNET_SPLITS.get(split_key, [split_key])
    visited: set[str] = set()
    count = 0
    for candidate in candidates:
        candidate = candidate.strip('/')
        candidate = candidate.strip('\\')
        if not candidate:
            continue
        if candidate in visited:
            continue
        visited.add(candidate)
        base = root / candidate
        if base.is_file():
            for record in _iter_json_records(base):
                if limit is not None and count >= limit:
                    return
                yield _prepare_document(record, base.stem)
                count += 1
            if count:
                return
        if not base.exists():
            continue
        if base.is_dir():
            path_candidates = list(sorted(base.glob("*.json"))) + list(sorted(base.glob("*.jsonl")))
            annotations = base / "annotations"
            if annotations.exists():
                path_candidates.extend(sorted(annotations.glob("*.json")))
                path_candidates.extend(sorted(annotations.glob("*.jsonl")))
            for path_obj in path_candidates:
                for record in _iter_json_records(path_obj):
                    if limit is not None and count >= limit:
                        return
                    yield _prepare_document(record, path_obj.stem)
                    count += 1
            if count:
                return
    if root.is_file():
        for record in _iter_json_records(root):
            if limit is not None and count >= limit:
                return
            yield _prepare_document(record, root.stem)
            count += 1

def _iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return iter(())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Fallback to JSON Lines if the file contains multiple JSON objects per line
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if isinstance(payload, Mapping):
        annotations = payload.get("annotations")
        if isinstance(annotations, Sequence):
            for item in annotations:
                if isinstance(item, Mapping):
                    yield dict(item)
            return
        yield dict(payload)
        return
    if isinstance(payload, Sequence):
        for item in payload:
            if isinstance(item, Mapping):
                yield dict(item)
        return
    return iter(())


def _prepare_document(raw: Mapping[str, Any], identifier: Optional[str]) -> Dict[str, Any]:
    document: Dict[str, Any] = dict(raw)
    doc_id = (
        document.get("doc_id")
        or document.get("document_id")
        or document.get("id")
        or identifier
        or ""
    )
    document["id"] = str(doc_id)
    document["doc_id"] = str(doc_id)
    width, height = _resolve_size(document)
    document["width"] = width
    document["height"] = height
    return document


def _extract_tokens(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates: Sequence[Any] = []
    for key in ("tokens", "words", "ocr_tokens", "ocr", "items", "segments", "lines"):
        value = document.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates = value
            break
        if isinstance(value, Mapping):
            nested = value.get("tokens") or value.get("words")
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                candidates = nested
                break
    tokens: List[Dict[str, Any]] = []
    for index, item in enumerate(candidates):
        if isinstance(item, Mapping):
            token = dict(item)
        else:
            token = {"text": str(item)}
        token_id = _coerce_int(token.get("id") or token.get("token_id") or token.get("idx"))
        if token_id is None:
            token_id = index
        token["id"] = int(token_id)
        tokens.append(token)
    return tokens


def _collect_regions(document: Mapping[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, int]]:
    candidates: Sequence[Any] = []
    for key in ("regions", "layout", "elements", "items", "annotations", "boxes"):
        value = document.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates = value
            break
    regions: Dict[int, Dict[str, Any]] = {}
    token_to_region: Dict[int, int] = {}
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            continue
        region = dict(item)
        region_id = _coerce_int(
            region.get("id")
            or region.get("region_id")
            or region.get("annotation_id")
            or region.get("idx")
        )
        if region_id is None:
            region_id = index
        region_id = int(region_id)
        tokens = region.get("token_ids")
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            tokens = region.get("tokens") or region.get("words")
        token_ids: List[int] = []
        if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
            for token in tokens:
                if isinstance(token, Mapping):
                    token_id = _coerce_int(token.get("id") or token.get("token_id") or token.get("idx"))
                else:
                    token_id = _coerce_int(token)
                if token_id is None:
                    continue
                token_id = int(token_id)
                token_ids.append(token_id)
                token_to_region[token_id] = region_id
        region["token_ids"] = token_ids
        region["label"] = _resolve_region_label(region)
        regions[region_id] = region
    return regions, token_to_region


def _resolve_region_label(region: Optional[Mapping[str, Any]]) -> str:
    if not region:
        return "other"
    for key in ("label", "category", "class", "type", "name", "kind"):
        value = region.get(key)
        if value is not None:
            return str(value)
    return "other"


def _resolve_token_text(token: Mapping[str, Any]) -> str:
    for key in ("text", "content", "value", "string", "span"):
        value = token.get(key)
        if value is not None:
            return str(value)
    return ""


def _resolve_page_index(source: Mapping[str, Any] | None, default: Optional[int] = None) -> int:
    if isinstance(source, Mapping):
        for key in ("page_index", "page", "page_id", "page_number", "page_num", "pageNo"):
            value = source.get(key)
            int_value = _coerce_int(value)
            if int_value is not None:
                return int(int_value)
    return int(default or 0)


def _resolve_size(document: Mapping[str, Any]) -> tuple[float, float]:
    for key in ("image_size", "img_size", "size"):
        size = document.get(key)
        if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) >= 2:
            width = float(size[0] or 1000)
            height = float(size[1] or 1000)
            return max(width, 1.0), max(height, 1.0)
    page = document.get("page_size")
    if isinstance(page, Mapping):
        width = page.get("width") or page.get("w")
        height = page.get("height") or page.get("h")
        if width and height:
            return max(float(width), 1.0), max(float(height), 1.0)
    width = float(document.get("width") or 1000)
    height = float(document.get("height") or 1000)
    return max(width, 1.0), max(height, 1.0)


def _normalise_box(box: Any, width: float, height: float) -> List[float]:
    if isinstance(box, Mapping):
        candidates = [
            box.get("x1") or box.get("left"),
            box.get("y1") or box.get("top"),
            box.get("x2") or box.get("right"),
            box.get("y2") or box.get("bottom"),
        ]
        if all(value is not None for value in candidates):
            box = candidates
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    left, top, right, bottom = [float(box[i]) for i in range(4)]
    return [left / w, top / h, right / w, bottom / h]


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_doclaynet_encoders",
    "build_doclaynet_registry",
    "doclaynet_contains_table",
    "doclaynet_fields",
    "load_doclaynet_dataset",
]
