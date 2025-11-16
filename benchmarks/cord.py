"""Utilities for loading and normalising the CORD receipt dataset."""

from __future__ import annotations

import json
import re
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
)

from nd_llm.encoders import Encoder, LayoutEncoder, TextEncoder
from nd_llm.registry import (
    FieldAdapter,
    FieldAdapterRegistry,
    LayoutAligner,
    Registry,
    quad_to_box,
)

try:  # pragma: no cover - optional dependency for the full dataset
    from datasets import load_dataset as _load_hf_dataset  # type: ignore
except Exception:  # pragma: no cover - fall back to the bundled sample
    _load_hf_dataset = None  # type: ignore[assignment]

__all__ = [
    "load_cord_dataset",
    "build_cord_registry",
    "build_cord_encoders",
    "build_cord_field_adapters",
    "cord_fields",
    "cord_amount_from_text",
    "cord_total_amount",
    "cord_high_total_label",
]

_DATA_DIR = Path(__file__).with_name("data")
_SAMPLE_PATH = _DATA_DIR.joinpath("cord_sample.jsonl")
_DATASET_NAME = "naver-clova-ix/cord-v2"
_AMOUNT_PATTERN = re.compile(r"-?\d[\d.,]*")
_CORD_FIELD_ADAPTERS: Optional[FieldAdapterRegistry] = None


def load_cord_dataset(
    *,
    split: str = "train",
    limit: Optional[int] = None,
    use_sample: bool = True,
    data_root: Optional[Path | str] = None,
    cache_dir: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Load the requested CORD split via Hugging Face or fall back to a bundled sample."""

    if data_root is not None:
        local_docs = _load_local_cord_documents(data_root, split=split, limit=limit)
        if local_docs:
            return local_docs
        raise FileNotFoundError(
            f"CORD data_root '{data_root}' does not contain any '{split}' JSON files"
        )

    documents: List[Dict[str, Any]] = []
    if use_sample:
        documents.extend(_load_sample(limit))
        if documents:
            return documents

    if _load_hf_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to download the CORD dataset. "
            "Install it with 'pip install datasets pillow' or pass use_sample=True."
        )

    dataset = _load_hf_dataset(
        _DATASET_NAME,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    for index, row in enumerate(dataset):
        document = _prepare_document(row)
        if not document.get("doc_id"):
            document["doc_id"] = f"cord-{split}-{index:05d}"
        documents.append(document)
        if limit is not None and len(documents) >= limit:
            break
    return documents


def build_cord_registry() -> Registry:
    """Return a registry describing the text, layout, and line-level fields."""

    registry = Registry()
    registry.add_field(
        "text",
        keys=["doc_id", "line_id", "token_id"],
        salience=True,
        modality="text",
    )
    registry.add_field(
        "layout",
        keys=["doc_id", "line_id", "token_id"],
        modality="layout",
    )
    registry.add_field(
        "line",
        keys=["doc_id", "line_id"],
        modality="entity",
    )
    registry.add_affinity("text", "layout", keys=["doc_id", "line_id", "token_id"])
    registry.add_affinity("line", "text", keys=["doc_id", "line_id"])
    registry.validate()
    return registry


def build_cord_encoders(
    registry: Registry,
    *,
    text_dim: int = 12,
    layout_dim: int = 8,
    line_dim: int = 6,
) -> Dict[str, Encoder]:
    """Register lightweight encoder stubs for the CORD registry."""

    encoders: Dict[str, Encoder] = {
        "text": TextEncoder(embedding_dim=text_dim),
        "layout": LayoutEncoder(embedding_dim=layout_dim),
        "line": TextEncoder(embedding_dim=line_dim),
    }
    for field, encoder in encoders.items():
        registry.register_encoder(field, encoder)
    return encoders


def build_cord_field_adapters() -> FieldAdapterRegistry:
    """Return field adapters that canonicalise CORD lines/words."""

    registry = FieldAdapterRegistry()
    aligner = LayoutAligner()
    registry.register(
        FieldAdapter(
            name="text",
            builder=_build_cord_text_entries,
            aligner=aligner,
        )
    )
    registry.register(
        FieldAdapter(
            name="layout",
            builder=_build_cord_layout_entries,
            aligner=aligner,
        )
    )
    registry.register(
        FieldAdapter(
            name="line",
            builder=_build_cord_line_entries,
            aligner=aligner,
        )
    )
    return registry


def cord_fields(
    document: Mapping[str, Any]
) -> Dict[str, List[MutableMapping[str, Any]]]:
    """Convert a normalised CORD document into registry-aligned field batches."""

    global _CORD_FIELD_ADAPTERS
    if _CORD_FIELD_ADAPTERS is None:
        _CORD_FIELD_ADAPTERS = build_cord_field_adapters()
    return _CORD_FIELD_ADAPTERS.transform(document)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _iter_cord_words(document: Mapping[str, Any]) -> Iterable[MutableMapping[str, Any]]:
    doc_id = str(document.get("doc_id") or document.get("id") or "")
    token_counter = 0
    for index, line in enumerate(document.get("lines", [])):
        line_id = _safe_int(line.get("line_id", line.get("group_id")), index)
        category = str(line.get("category", "other"))
        group_id = _safe_int(line.get("group_id"), line_id)
        sub_group_id = _safe_int(line.get("sub_group_id"), 0)
        for word in line.get("words", []):
            token_id = word.get("token_id")
            if token_id is None:
                token_id = token_counter
            token_counter += 1
            entry: MutableMapping[str, Any] = {
                "doc_id": doc_id,
                "line_id": line_id,
                "token_id": _safe_int(token_id, token_counter),
                "text": str(word.get("text", "")),
                "category": category,
                "group_id": group_id,
                "sub_group_id": sub_group_id,
                "quad": word.get("quad") or word.get("coords"),
            }
            yield entry


def _build_cord_text_entries(
    document: Mapping[str, Any]
) -> Iterable[MutableMapping[str, Any]]:
    return _iter_cord_words(document)


def _build_cord_layout_entries(
    document: Mapping[str, Any]
) -> Iterable[MutableMapping[str, Any]]:
    for word in _iter_cord_words(document):
        entry: MutableMapping[str, Any] = {
            "doc_id": word["doc_id"],
            "line_id": word["line_id"],
            "token_id": word["token_id"],
            "category": word.get("category"),
            "quad": word.get("quad"),
        }
        yield entry


def _build_cord_line_entries(
    document: Mapping[str, Any]
) -> Iterable[MutableMapping[str, Any]]:
    doc_id = str(document.get("doc_id") or document.get("id") or "")
    for index, line in enumerate(document.get("lines", [])):
        words = line.get("words", [])
        text_value = " ".join(
            str(word.get("text", "")).strip() for word in words if word.get("text")
        ).strip()
        entry: MutableMapping[str, Any] = {
            "doc_id": doc_id,
            "line_id": _safe_int(line.get("line_id", line.get("group_id")), index),
            "category": str(line.get("category", "other")),
            "group_id": _safe_int(line.get("group_id"), index),
            "sub_group_id": _safe_int(line.get("sub_group_id"), 0),
            "text": text_value,
            "token_count": len(words),
            "quad": _line_quad(line),
        }
        yield entry


def _line_quad(line: Mapping[str, Any]) -> Optional[List[float]]:
    quads: List[List[float]] = []
    for word in line.get("words", []):
        quad = word.get("quad")
        if quad is not None:
            quads.append(quad_to_box(quad))
    if quads:
        min_x = min(box[0] for box in quads)
        min_y = min(box[1] for box in quads)
        max_x = max(box[2] for box in quads)
        max_y = max(box[3] for box in quads)
        return [min_x, min_y, max_x, max_y]
    quad = line.get("quad")
    if quad is not None:
        return quad_to_box(quad)
    coords = line.get("coords")
    if isinstance(coords, Sequence):
        return [float(value) for value in coords[:4]]
    return [0.0, 0.0, 0.0, 0.0]


def cord_amount_from_text(value: Any) -> float:
    """Best-effort extraction of a numeric amount from a free-form token."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    candidates = _AMOUNT_PATTERN.findall(text.replace(" ", ""))
    best = 0.0
    for candidate in candidates:
        cleaned = candidate.replace(",", "")
        if cleaned.count(".") > 1:
            cleaned = cleaned.replace(".", "", cleaned.count(".") - 1)
        try:
            amount = float(cleaned)
        except Exception:
            continue
        if abs(amount) > abs(best):
            best = amount
    return best


def cord_total_amount(document: Mapping[str, Any]) -> float:
    """Return the parsed total amount for a document."""

    if "total_amount" in document:
        try:
            return float(document["total_amount"])
        except Exception:
            pass
    total = document.get("total") or document.get("totals") or {}
    sub_total = document.get("sub_total") or document.get("subtotal") or {}
    for key in ("total_price", "cashprice", "creditcard_price", "total"):
        if key in total:
            amount = cord_amount_from_text(total[key])
            if amount:
                return amount
    for key in ("subtotal_price", "sum"):
        if key in sub_total:
            amount = cord_amount_from_text(sub_total[key])
            if amount:
                return amount
    return 0.0


def cord_high_total_label(document: Mapping[str, Any], *, threshold: float) -> bool:
    """Binary label for whether a receipt exceeds the provided total threshold."""

    return cord_total_amount(document) >= float(threshold)


def _load_sample(limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    if not _SAMPLE_PATH.exists():
        return
    count = 0
    with _SAMPLE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield _prepare_document(data)
            count += 1
            if limit is not None and count >= limit:
                return


def _load_local_cord_documents(
    root: Path | str,
    *,
    split: str,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    base = Path(root).expanduser().resolve()
    roots = _resolve_local_roots(base)
    documents: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for dataset_root in roots:
        for doc in _iter_local_split(dataset_root, split):
            doc_id = str(doc.get("doc_id") or "")
            if not doc_id:
                doc_id = f"{dataset_root.name}-{split}-{len(documents):05d}"
                doc["doc_id"] = doc_id
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            documents.append(doc)
            if limit is not None and len(documents) >= limit:
                return documents
    return documents


def _resolve_local_roots(base: Path) -> List[Path]:
    def _is_cord_root(candidate: Path) -> bool:
        return any((candidate / part).is_dir() for part in ("train", "test", "dev"))

    candidates: List[Path] = []
    if _is_cord_root(base):
        return [base]
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if not name.startswith("cord"):
            continue
        if _is_cord_root(child):
            candidates.append(child)
    return candidates


def _iter_local_split(root: Path, split: str) -> Iterator[Dict[str, Any]]:
    split_dir = root / split
    json_dir = split_dir / "json"
    if not json_dir.is_dir():
        json_dir = split_dir
    if not json_dir.is_dir():
        return
    for path in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        document = _prepare_document(data)
        document.setdefault("doc_id", path.stem)
        yield document


def _prepare_document(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if ("lines" in raw or "valid_line" in raw) and "width" in raw and "height" in raw:
        return _finalise_document(raw)

    payload: Mapping[str, Any]
    ground_truth = raw.get("ground_truth")
    if isinstance(ground_truth, str):
        payload = json.loads(ground_truth)
    elif isinstance(ground_truth, Mapping):
        payload = ground_truth
    else:
        payload = raw
    meta = payload.get("meta", {})

    image_size = meta.get("image_size") or {}
    width = image_size.get("width")
    height = image_size.get("height")
    if (not width or not height) and "image" in raw:
        image = raw["image"]
        try:
            width = getattr(image, "width")
            height = getattr(image, "height")
        except Exception:
            try:
                size = getattr(image, "size")
                width, height = size
            except Exception:
                width = width or 1000
                height = height or 1400

    converted: Dict[str, Any] = {
        "doc_id": raw.get("doc_id") or raw.get("id") or meta.get("image_id") or "",
        "width": int(width or 1000),
        "height": int(height or 1400),
        "lines": payload.get("valid_line") or [],
        "menu": payload.get("gt_parse", {}).get("menu", []),
        "sub_total": payload.get("gt_parse", {}).get("sub_total", {}),
        "total": payload.get("gt_parse", {}).get("total", {}),
        "metadata": {
            "split": meta.get("split") or raw.get("split"),
            "version": meta.get("version"),
        },
    }
    return _finalise_document(converted)


def _finalise_document(raw: Mapping[str, Any]) -> Dict[str, Any]:
    doc_id = str(raw.get("doc_id") or raw.get("id") or "")
    width = max(int(raw.get("width", 0) or 1), 1)
    height = max(int(raw.get("height", 0) or 1), 1)
    metadata = dict(raw.get("metadata") or {})
    metadata.setdefault("source", raw.get("source", "cord"))

    lines = _prepare_lines(raw.get("lines") or raw.get("valid_line") or [])
    document = {
        "doc_id": doc_id,
        "width": width,
        "height": height,
        "lines": lines,
        "menu": list(raw.get("menu") or []),
        "sub_total": dict(raw.get("sub_total") or raw.get("subtotal") or {}),
        "total": dict(raw.get("total") or raw.get("totals") or {}),
        "metadata": metadata,
    }
    document["total_amount"] = cord_total_amount(document)
    return document


def _prepare_lines(lines: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    token_counter = 0
    for index, line in enumerate(lines):
        line_id = line.get("line_id")
        if line_id is None:
            line_id = line.get("group_id")
        if line_id is None:
            line_id = index
        category = str(line.get("category", "other"))
        group_id = int(line.get("group_id", line_id))
        sub_group_id = int(line.get("sub_group_id", 0))

        prepared_words = []
        for word in line.get("words", []):
            token_id = word.get("token_id")
            if token_id is None:
                token_id = token_counter
            token_counter += 1
            prepared_words.append(
                {
                    "token_id": int(token_id),
                    "text": str(word.get("text", "")),
                    "quad": _normalise_quad(word.get("quad")),
                    "is_key": int(word.get("is_key", 0)),
                }
            )

        prepared.append(
            {
                "line_id": int(line_id),
                "category": category,
                "group_id": group_id,
                "sub_group_id": sub_group_id,
                "words": prepared_words,
            }
        )
    return prepared


def _normalise_quad(quad: Any) -> Dict[str, float]:
    if isinstance(quad, Mapping):
        return {
            "x1": float(quad.get("x1", 0.0)),
            "y1": float(quad.get("y1", 0.0)),
            "x2": float(quad.get("x2", quad.get("x1", 0.0))),
            "y2": float(quad.get("y2", quad.get("y1", 0.0))),
            "x3": float(quad.get("x3", quad.get("x2", 0.0))),
            "y3": float(quad.get("y3", quad.get("y2", 0.0))),
            "x4": float(quad.get("x4", quad.get("x1", 0.0))),
            "y4": float(quad.get("y4", quad.get("y3", 0.0))),
        }
    if isinstance(quad, Sequence) and len(quad) >= 8:
        return {
            "x1": float(quad[0]),
            "y1": float(quad[1]),
            "x2": float(quad[2]),
            "y2": float(quad[3]),
            "x3": float(quad[4]),
            "y3": float(quad[5]),
            "x4": float(quad[6]),
            "y4": float(quad[7]),
        }
    return {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 0.0,
        "y2": 0.0,
        "x3": 0.0,
        "y3": 0.0,
        "x4": 0.0,
        "y4": 0.0,
    }
