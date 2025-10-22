"""Utilities for loading and normalising the FUNSD form-understanding dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from nd_llm.encoders import Encoder, LayoutEncoder, TextEncoder
from nd_llm.registry import Registry

_SAMPLE_PATH = Path(__file__).with_name("data").joinpath("funsd_sample.jsonl")

_FUNSD_SPLITS: Dict[str, List[str]] = {
    "train": ["train", "training", "training_data"],
    "training": ["train", "training", "training_data"],
    "test": ["test", "testing", "testing_data"],
    "testing": ["test", "testing", "testing_data"],
    "validation": ["validation", "valid", "val", "dev", "testing", "testing_data"],
    "val": ["validation", "valid", "val", "dev", "testing", "testing_data"],
}


def load_funsd_dataset(
    root: Optional[Path | str] = None,
    *,
    split: str = "train",
    limit: Optional[int] = None,
    use_sample: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Load FUNSD documents from ``root`` or fall back to the bundled sample."""

    documents: List[Dict[str, Any]] = []
    if use_sample or root is None:
        documents = list(_load_sample(limit))
    else:
        path = Path(root)
        if not path.exists():
            raise FileNotFoundError(f"FUNSD root directory '{path}' does not exist")
        documents = list(_load_from_directory(path, split=split, limit=limit))

    if not documents and not use_sample and root is None:
        documents = list(_load_sample(limit))
    return documents


def build_funsd_registry() -> Registry:
    """Return a registry describing FUNSD token, layout and entity fields."""

    registry = Registry()
    registry.add_field("text", keys=["doc_id", "token_id"], salience=True, modality="text")
    registry.add_field("layout", keys=["doc_id", "token_id"], modality="layout")
    registry.add_field("entity", keys=["doc_id", "entity_id"], modality="entity")
    registry.add_affinity("text", "layout", keys=["doc_id", "token_id"])
    registry.add_affinity("entity", "text", keys=["doc_id"])
    registry.validate()
    return registry


def build_funsd_encoders(
    registry: Registry,
    *,
    text_dim: int = 8,
    layout_dim: int = 6,
    entity_dim: int = 4,
) -> Dict[str, Encoder]:
    """Register simple encoder stubs for the FUNSD fields."""

    encoders: Dict[str, Encoder] = {
        "text": TextEncoder(embedding_dim=text_dim),
        "layout": LayoutEncoder(embedding_dim=layout_dim),
        "entity": TextEncoder(embedding_dim=entity_dim),
    }
    for field, encoder in encoders.items():
        registry.register_encoder(field, encoder)
    return encoders


def funsd_fields(document: Mapping[str, Any]) -> Dict[str, List[MutableMapping[str, Any]]]:
    """Convert a FUNSD document into registry-aligned field batches."""

    doc_id = str(document.get("doc_id") or document.get("id") or "")
    width, height = _resolve_size(document)
    text_field: List[MutableMapping[str, Any]] = []
    layout_field: List[MutableMapping[str, Any]] = []
    entity_field: List[MutableMapping[str, Any]] = []

    for entity_index, raw_entity in enumerate(document.get("form", [])):
        entity_id = int(raw_entity.get("id", entity_index))
        label = str(raw_entity.get("label", "other"))
        raw_text = raw_entity.get("text")
        if not raw_text:
            raw_text = " ".join(str(word.get("text", "")) for word in raw_entity.get("words", []))
        raw_text = str(raw_text).strip()

        token_ids: List[int] = []
        for word_index, raw_word in enumerate(raw_entity.get("words", [])):
            token_id = raw_word.get("id")
            if token_id is None:
                token_id = len(text_field)
            token_id = int(token_id)
            token_ids.append(token_id)
            token_text = str(raw_word.get("text", ""))
            text_field.append(
                {
                    "doc_id": doc_id,
                    "token_id": token_id,
                    "text": token_text,
                    "entity_id": entity_id,
                    "entity_label": label,
                    "is_answer": label.lower() == "answer",
                }
            )

            bbox = raw_word.get("box") or raw_word.get("bbox")
            norm_box = _normalise_box(bbox, width, height)
            layout_field.append(
                {
                    "doc_id": doc_id,
                    "token_id": token_id,
                    "xyxy": norm_box,
                    "entity_id": entity_id,
                }
            )

        entity_field.append(
            {
                "doc_id": doc_id,
                "entity_id": entity_id,
                "label": label,
                "text": raw_text,
                "token_ids": token_ids,
            }
        )

    return {"text": text_field, "layout": layout_field, "entity": entity_field}


def funsd_numeric_answer_label(document: Mapping[str, Any]) -> bool:
    """Return ``True`` if an answer entity contains a digit."""

    for entity in document.get("form", []):
        label = str(entity.get("label", "")).lower()
        if label != "answer":
            continue
        raw_text = entity.get("text")
        if not raw_text:
            raw_text = " ".join(str(word.get("text", "")) for word in entity.get("words", []))
        if any(char.isdigit() for char in str(raw_text)):
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
    candidates = _FUNSD_SPLITS.get(split_key, [split_key])
    visited = set()
    for candidate in candidates:
        candidate = candidate.strip("/")
        if not candidate:
            continue
        if candidate in visited:
            continue
        visited.add(candidate)
        base = root / candidate
        if not base.exists():
            continue
        annotations_dir = base / "annotations"
        if not annotations_dir.exists():
            continue
        count = 0
        for path in sorted(annotations_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            document = _prepare_document(data, path.stem)
            yield document
            count += 1
            if limit is not None and count >= limit:
                return
        if count:
            return
    if split_key in {"validation", "val"}:
        yield from _load_from_directory(root, split="test", limit=limit)


def _prepare_document(raw: Mapping[str, Any], identifier: Optional[str]) -> Dict[str, Any]:
    document: Dict[str, Any] = dict(raw)
    doc_id = document.get("id") or document.get("doc_id") or identifier or ""
    document["id"] = str(doc_id)
    document["doc_id"] = str(doc_id)
    if "form" not in document:
        if "annotations" in document and isinstance(document["annotations"], Sequence):
            document["form"] = list(document["annotations"])
        else:
            document["form"] = []
    width, height = _resolve_size(document)
    document["width"] = width
    document["height"] = height
    return document


def _resolve_size(document: Mapping[str, Any]) -> tuple[float, float]:
    for key in ("image_size", "img_size", "size"):
        size = document.get(key)
        if isinstance(size, Sequence) and len(size) >= 2:
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
            box.get("x1"),
            box.get("y1"),
            box.get("x2"),
            box.get("y2"),
        ]
        if all(value is not None for value in candidates):
            box = candidates
    if not isinstance(box, Sequence) or len(box) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    left, top, right, bottom = [float(box[i]) for i in range(4)]
    return [left / w, top / h, right / w, bottom / h]


__all__ = [
    "build_funsd_encoders",
    "build_funsd_registry",
    "funsd_fields",
    "funsd_numeric_answer_label",
    "load_funsd_dataset",
]
