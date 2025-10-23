"""File-backed short-term memory for tensor persistence."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import zlib
from array import array
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
    TYPE_CHECKING,
    Union,
)

from nd_llm.utils.config import STMConfig

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any  # type: ignore[assignment]

try:  # pragma: no cover - numpy is optional in the test environment
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - fallback when numpy is missing
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - torch is optional in the test environment
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None  # type: ignore[assignment]

_NUMPY_AVAILABLE = _np is not None
_TORCH_AVAILABLE = torch is not None

TensorLike = Any


class STM:
    """A lightweight file-backed short-term memory for tensors and metadata."""

    def __init__(self, config: STMConfig) -> None:
        self._config = config
        self._storage_dir = Path(config.storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._storage_dir / config.index_filename
        self._lock = threading.RLock()
        self._index: Dict[str, MutableMapping[str, Any]] = self._load_index()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @classmethod
    def from_path(
        cls,
        storage_dir: Union[str, Path],
        *,
        index_filename: str = "index.json",
    ) -> "STM":
        """Instantiate an :class:`STM` directly from storage parameters."""

        config = STMConfig(storage_dir=storage_dir, index_filename=index_filename)
        return cls(config)

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def append(
        self,
        key: str,
        tensor: TensorLike,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist a tensor and its metadata under the provided key."""

        if not key:
            raise ValueError("key must be a non-empty string")

        payload_bytes, shape, length = self._prepare_payload(tensor)
        payload_metadata = self._normalize_metadata(metadata)
        entry = self._build_index_entry(key, shape, length, payload_metadata)

        with self._lock:
            if key in self._index:
                raise KeyError(f"Key '{key}' already exists in STM index")

            tensor_path = self._storage_dir / entry["tensor_file"]
            tmp_path = tensor_path.with_suffix(".tmp")
            compressed = zlib.compress(payload_bytes)
            with tmp_path.open("wb") as fh:
                fh.write(compressed)
            tmp_path.replace(tensor_path)

            self._index[key] = entry
            self._save_index()

    def retrieve(self, key: str) -> tuple[Any, Dict[str, Any]]:
        """Load the tensor and metadata associated with ``key``."""

        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                raise KeyError(f"Key '{key}' not found in STM index")
            tensor_path = self._storage_dir / entry["tensor_file"]

        if not tensor_path.exists():
            raise FileNotFoundError(f"Tensor payload missing for key '{key}' at {tensor_path}")

        compressed = tensor_path.read_bytes()
        buffer = zlib.decompress(compressed)

        stored_order = entry.get("byteorder")
        if stored_order and stored_order != sys.byteorder:
            raise RuntimeError(
                "Tensor payload byte order mismatch: "
                f"stored={stored_order} current={sys.byteorder}"
            )

        values_array = array("d")
        values_array.frombytes(buffer)
        flat_values = values_array.tolist()

        expected_length = entry.get("length")
        if expected_length is not None and expected_length != len(flat_values):
            raise ValueError(
                f"Tensor payload length mismatch for key '{key}': "
                f"expected {expected_length}, found {len(flat_values)}"
            )

        shape = entry.get("shape", [])
        tensor_obj: Any
        if _NUMPY_AVAILABLE:
            tensor_obj = _np.array(flat_values, dtype=_np.float64)  # type: ignore[union-attr]
            if shape:
                tensor_obj = tensor_obj.reshape(shape)
        else:
            iterator = iter(flat_values)
            tensor_obj = self._reshape_from_iter(iterator, list(shape))
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    f"Tensor payload for key '{key}' contains extra values beyond the declared shape"
                )

        metadata = json.loads(json.dumps(entry.get("metadata", {})))
        return tensor_obj, metadata

    def list_keys(self) -> Sequence[str]:
        with self._lock:
            return list(self._index.keys())

    def get_index_entry(self, key: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                raise KeyError(f"Key '{key}' not found in STM index")
            return json.loads(json.dumps(entry))

    def query(
        self,
        *,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> list[tuple[str, Dict[str, Any]]]:
        with self._lock:
            items = list(self._index.items())

        results: list[tuple[str, Dict[str, Any]]] = []
        for key, entry in items:
            metadata = entry.get("metadata", {})
            if metadata_filter and not self._metadata_matches(metadata, metadata_filter):
                continue
            results.append((key, json.loads(json.dumps(entry))))
            if limit is not None and len(results) >= limit:
                break
        return results

    def list_by_alignment(self, alignment_key: str, limit: Optional[int] = None) -> Sequence[str]:
        matches = self.query(metadata_filter={"alignment_key": alignment_key}, limit=limit)
        return [key for key, _ in matches]

    def list_by_task(self, task: str, limit: Optional[int] = None) -> Sequence[str]:
        matches = self.query(metadata_filter={"task": task}, limit=limit)
        return [key for key, _ in matches]

    def list_by_layout(self, layout_signature: str, limit: Optional[int] = None) -> Sequence[str]:
        matches = self.query(metadata_filter={"layout_signature": layout_signature}, limit=limit)
        return [key for key, _ in matches]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._index

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build_index_entry(
        self,
        key: str,
        shape: Sequence[int],
        length: int,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        tensor_filename = self._tensor_filename(key)
        entry = {
            "tensor_file": tensor_filename,
            "shape": list(shape),
            "dtype": "float64",
            "length": int(length),
            "byteorder": sys.byteorder,
            "compression": "zlib",
            "metadata": metadata,
        }
        entry.update(self._derive_index_annotations(metadata))
        return entry

    def _tensor_filename(self, key: str) -> str:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        return f"{safe_key}_{digest}.bin"

    def _prepare_payload(self, tensor: TensorLike) -> tuple[bytes, Sequence[int], int]:
        nested = self._to_nested_structure(tensor)
        shape = self._infer_shape(nested)
        flat = self._flatten_nested(nested)
        payload_array = array("d", flat)
        return payload_array.tobytes(), shape, len(flat)

    def _to_nested_structure(self, tensor: TensorLike) -> Any:
        if _NUMPY_AVAILABLE and _np is not None and isinstance(tensor, _np.ndarray):
            return tensor.tolist()
        if _TORCH_AVAILABLE and torch is not None and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().tolist()
        if isinstance(tensor, (list, tuple)):
            return [self._to_nested_structure(item) for item in tensor]
        if isinstance(tensor, Iterable) and not isinstance(tensor, (str, bytes, bytearray)):
            return [self._to_nested_structure(item) for item in list(tensor)]
        if isinstance(tensor, (int, float)):
            return float(tensor)
        raise TypeError(f"Unsupported tensor type: {type(tensor)!r}")

    def _infer_shape(self, value: Any) -> list[int]:
        if isinstance(value, list):
            length = len(value)
            if length == 0:
                return [0]
            first_shape = self._infer_shape(value[0])
            for item in value[1:]:
                if self._infer_shape(item) != first_shape:
                    raise ValueError("Inconsistent tensor shape for STM payload")
            return [length] + first_shape
        return []

    def _flatten_nested(self, value: Any) -> list[float]:
        if isinstance(value, list):
            flattened: list[float] = []
            for item in value:
                flattened.extend(self._flatten_nested(item))
            return flattened
        return [float(value)]

    def _reshape_from_iter(self, iterator: Iterator[float], shape: Sequence[int]) -> Any:
        if not shape:
            try:
                return next(iterator)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise ValueError("Tensor payload is shorter than expected") from exc
        size = shape[0]
        remainder = shape[1:]
        result = [self._reshape_from_iter(iterator, remainder) for _ in range(size)]
        return result

    def _normalize_metadata(self, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if metadata is None:
            metadata_dict: Dict[str, Any] = {}
        else:
            metadata_dict = dict(metadata)
        json.dumps(metadata_dict)
        return metadata_dict

    def _metadata_matches(
        self,
        metadata: Mapping[str, Any],
        metadata_filter: Mapping[str, Any],
    ) -> bool:
        for dotted_key, expected in metadata_filter.items():
            value: Any = metadata
            for part in str(dotted_key).split("."):
                if isinstance(value, Mapping) and part in value:
                    value = value[part]
                else:
                    value = None
                    break

            if isinstance(expected, (list, tuple, set, frozenset)):
                if value not in expected:
                    return False
            elif expected is None:
                if value is not None:
                    return False
            else:
                if value != expected:
                    return False
        return True

    def _derive_index_annotations(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        annotations: Dict[str, Any] = {}
        compression_data = metadata.get("compression") if isinstance(metadata, Mapping) else None
        if isinstance(compression_data, Mapping):
            summary = compression_data.get("summary")
            if isinstance(summary, Mapping):
                compression_summary = {}
                ratio = summary.get("compression_ratio")
                if isinstance(ratio, (int, float)):
                    compression_summary["compression_ratio"] = float(ratio)
                retained = summary.get("tokens_retained")
                if isinstance(retained, (int, float)):
                    compression_summary["tokens_retained"] = float(retained)
                total = summary.get("tokens_total")
                if isinstance(total, (int, float)):
                    compression_summary["tokens_total"] = float(total)
                if compression_summary:
                    annotations["compression_summary"] = compression_summary

            telemetry = compression_data.get("telemetry")
            if isinstance(telemetry, Mapping):
                budget = telemetry.get("budget")
                if isinstance(budget, (int, float)):
                    annotations["compression_budget"] = float(budget)

            fields_info = compression_data.get("fields")
            if isinstance(fields_info, Mapping):
                names = fields_info.get("names")
                if isinstance(names, (list, tuple, set)):
                    annotations["compression_fields"] = [str(name) for name in names]
                counts = fields_info.get("counts")
                if isinstance(counts, Mapping):
                    annotations["compression_field_counts"] = {
                        str(field): int(value)
                        for field, value in counts.items()
                        if isinstance(value, (int, float))
                    }

            idx_cells = compression_data.get("idx_cells")
            if isinstance(idx_cells, Mapping):
                def _normalise_index_map(value: Any) -> Dict[str, List[int]]:
                    if not isinstance(value, Mapping):
                        return {}
                    result: Dict[str, List[int]] = {}
                    for field, entries in value.items():
                        normalised: List[int] = []
                        if isinstance(entries, Mapping):
                            entries = entries.values()
                        for raw in (entries if isinstance(entries, (list, tuple, set, frozenset)) else [entries]):
                            try:
                                normalised.append(int(raw))
                            except (TypeError, ValueError):
                                continue
                        if normalised:
                            result[str(field)] = normalised
                    return result

                kept_map = _normalise_index_map(idx_cells.get("kept"))
                if kept_map:
                    annotations["idx_cells_kept"] = kept_map

                dropped_map = _normalise_index_map(idx_cells.get("dropped"))
                if dropped_map:
                    annotations["idx_cells_dropped"] = dropped_map

                canonical_raw = idx_cells.get("canonical")
                canonical_map: Dict[str, List[str]] = {}
                if isinstance(canonical_raw, Mapping):
                    for field, values in canonical_raw.items():
                        if isinstance(values, Mapping):
                            values = values.values()
                        if isinstance(values, (list, tuple, set, frozenset)):
                            serialised = [str(item) for item in values if item is not None]
                        elif values is None:
                            serialised = []
                        else:
                            serialised = [str(values)]
                        if serialised:
                            canonical_map[str(field)] = serialised
                if canonical_map:
                    annotations["canonical_cells"] = canonical_map
                    signature = _canonical_layout_signature(canonical_map)
                    if signature:
                        annotations.setdefault("layout_signature", signature)

        if isinstance(metadata, Mapping):
            task = metadata.get("task")
            if isinstance(task, str):
                annotations["task"] = task
            layout_signature_value = metadata.get("layout_signature")
            if isinstance(layout_signature_value, str):
                annotations.setdefault("layout_signature", layout_signature_value)
            k_value = metadata.get("K")
            if isinstance(k_value, (int, float)):
                annotations["K"] = float(k_value)
            mi_lb = metadata.get("mi_lb")
            if isinstance(mi_lb, (int, float)):
                annotations["mi_lb"] = float(mi_lb)

        alignment_key = metadata.get("alignment_key") if isinstance(metadata, Mapping) else None
        if isinstance(alignment_key, str):
            annotations["alignment_key"] = alignment_key

        return annotations

    def _load_index(self) -> Dict[str, MutableMapping[str, Any]]:
        if not self._index_path.exists():
            return {}

        with self._index_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("STM index file must contain a JSON object")
        return {str(key): dict(value) for key, value in data.items()}

    def _save_index(self) -> None:
        tmp_path = self._index_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self._index, fh, indent=2, sort_keys=True)
        tmp_path.replace(self._index_path)


def _canonical_layout_signature(canonical: Mapping[str, Sequence[Any]]) -> Optional[str]:
    parts: list[str] = []
    for field in sorted(canonical):
        values = canonical[field]
        if isinstance(values, (list, tuple, set, frozenset)):
            serialised = [str(item) for item in values if item is not None]
        elif values is None:
            serialised = []
        else:
            serialised = [str(values)]
        if serialised:
            parts.append(f"{field}:{','.join(serialised)}")
    if not parts:
        return None
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


__all__ = ["STM", "TensorLike"]
