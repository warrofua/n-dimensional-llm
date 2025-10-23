"""Field packing helpers for quick-start workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, Iterator, List, Optional, Sequence


_PREFERRED_VALUE_KEYS: Sequence[str] = (
    "value",
    "values",
    "text",
    "content",
    "payload",
    "data",
    "xyxy",
)


class PackedFields(Mapping[str, List[Any]]):
    """Dictionary-like container returned by :func:`pack_fields`.

    The object retains the packed per-field payloads and the ordered
    alignment keys that were discovered (or explicitly provided) during
    normalization.  Consumers can index ``PackedFields`` directly to obtain
    the per-field payload sequences or call :meth:`to_field_batches` to
    obtain encoder-ready value batches.
    """

    def __init__(
        self,
        payloads: Dict[str, List[Any]],
        key_rows: List[Dict[Any, Any]],
        key_order: Sequence[Any],
    ) -> None:
        self._payloads = {name: list(items) for name, items in payloads.items()}
        self._key_rows = [dict(row) for row in key_rows]
        self._key_order = list(key_order)

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> List[Any]:  # type: ignore[override]
        return self._payloads[key]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self._payloads)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._payloads)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def field_names(self) -> List[str]:
        """Return the list of packed field names."""

        return list(self._payloads)

    @property
    def key_names(self) -> List[Any]:
        """Return the discovered (or provided) alignment key names."""

        return list(self._key_order)

    @property
    def key_rows(self) -> List[Dict[Any, Any]]:
        """Return a defensive copy of the alignment key rows."""

        return [dict(row) for row in self._key_rows]

    def to_field_batches(self, value_keys: Optional[Mapping[str, Any]] = None) -> Dict[str, List[Any]]:
        """Convert the packed payloads into encoder-ready batches.

        Parameters
        ----------
        value_keys:
            Optional mapping of field name to the key that should be treated
            as the primary value for that field.  When omitted the helper
            falls back to a set of common heuristics (``"value"``,
            ``"text"``, ``"xyxy"`` …) and, failing that, to the entire
            payload mapping.
        """

        batches: Dict[str, List[Any]] = {}
        for field, entries in self._payloads.items():
            preferred = None
            if value_keys and field in value_keys:
                preferred = value_keys[field]
            batches[field] = [
                _select_value(field, entry, preferred) for entry in entries
            ]
        return batches

    def as_dict(self) -> Dict[str, List[Any]]:
        """Return a deep copy of the packed payload mapping."""

        return {field: [_copy_entry(entry) for entry in entries] for field, entries in self._payloads.items()}

    def with_keys(self) -> Dict[str, List[Any]]:
        """Return payload rows merged with their corresponding key columns."""

        combined: Dict[str, List[Any]] = {}
        key_rows = self._key_rows
        for field, entries in self._payloads.items():
            merged_entries: List[Any] = []
            for idx, entry in enumerate(entries):
                merged_entries.append(self._merge_entry_with_keys(entry, key_rows, idx))
            combined[field] = merged_entries
        return combined

    @staticmethod
    def _merge_entry_with_keys(entry: Any, key_rows: Sequence[Dict[Any, Any]], index: int) -> Any:
        payload = _copy_entry(entry)
        if index >= len(key_rows):
            return payload
        key_row = key_rows[index]
        if not key_row:
            return payload
        merged = dict(key_row)
        if isinstance(payload, Mapping):
            merged.update(payload)
            return merged
        merged["value"] = payload
        return merged

    def __repr__(self) -> str:
        return f"PackedFields(fields={list(self._payloads)}, keys={self._key_order})"


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------

def pack_fields(*, keys: Optional[Sequence[Any]] = None, **field_batches: Iterable[Any]) -> PackedFields:
    """Normalize keyword field batches into encoder-friendly payloads.

    The helper accepts keyword arguments mapping field names to iterables of
    per-row payload dictionaries (or other values).  It verifies that the
    batches share a common length and that any declared key columns are
    present across fields.  Alignment keys are either inferred from the
    shared dictionary keys or supplied explicitly via ``keys``.
    """

    if not field_batches:
        raise ValueError("pack_fields requires at least one field sequence")

    materialized: Dict[str, List[Any]] = {}
    for field, raw in field_batches.items():
        materialized[field] = _materialize_entries(raw)

    lengths = {len(entries) for entries in materialized.values()}
    if len(lengths) != 1:
        raise ValueError("all field batches must contain the same number of rows")

    length = lengths.pop() if lengths else 0

    if keys is not None:
        key_order = list(keys)
        if len(set(key_order)) != len(key_order):
            raise ValueError("duplicate key names detected in 'keys' argument")
        key_set = set(key_order)
    else:
        key_set, per_field_keys = _collect_field_keys(materialized)
        _validate_key_alignment(per_field_keys, key_set)
        key_order = _determine_key_order(materialized, key_set)

    _validate_key_presence(materialized, key_order)

    key_rows = _build_key_rows(materialized, key_order, length)
    payloads = _strip_key_columns(materialized, key_set)

    return PackedFields(payloads=payloads, key_rows=key_rows, key_order=key_order)


# ----------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------

def _materialize_entries(raw: Iterable[Any]) -> List[Any]:
    if isinstance(raw, Mapping):
        return [dict(raw)]
    if isinstance(raw, (str, bytes, bytearray)):
        return [raw]
    try:
        iterator = iter(raw)
    except TypeError:
        return [raw]

    entries: List[Any] = []
    for item in iterator:
        if isinstance(item, Mapping):
            entries.append(dict(item))
        else:
            entries.append(item)
    return entries


def _collect_field_keys(
    materialized: Mapping[str, List[Any]]
) -> tuple[set[Any], Dict[str, set[Any]]]:
    common_keys: Optional[set[Any]] = None
    per_field: Dict[str, set[Any]] = {}

    for field, entries in materialized.items():
        field_keys: Optional[set[Any]] = None
        for item in entries:
            if isinstance(item, Mapping):
                item_keys = set(item.keys())
                if field_keys is None:
                    field_keys = set(item_keys)
                else:
                    field_keys &= item_keys
        if field_keys is None:
            field_keys = set()
        per_field[field] = field_keys
        if common_keys is None:
            common_keys = set(field_keys)
        else:
            common_keys &= field_keys

    return common_keys or set(), per_field


def _validate_key_alignment(per_field_keys: Mapping[str, set[Any]], key_set: set[Any]) -> None:
    if not per_field_keys:
        return

    candidate_keys: set[Any] = set(key_set)
    for keys in per_field_keys.values():
        for key in keys:
            if _looks_like_alignment_key(key):
                candidate_keys.add(key)

    if not candidate_keys:
        return

    missing_descriptions: List[str] = []
    for field, keys in per_field_keys.items():
        missing = sorted(str(key) for key in candidate_keys - keys)
        if missing:
            missing_descriptions.append(f"{field}: {', '.join(missing)}")

    if missing_descriptions:
        detail = "; ".join(missing_descriptions)
        raise ValueError(f"field batches are missing alignment keys: {detail}")


def _looks_like_alignment_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    suffixes = ("id", "_id", "key", "_key", "idx", "_idx", "index")
    return any(lowered.endswith(suffix) for suffix in suffixes)


def _determine_key_order(materialized: Mapping[str, List[Any]], key_set: set[Any]) -> List[Any]:
    if not key_set:
        return []

    key_order: List[Any] = []
    for entries in materialized.values():
        for item in entries:
            if isinstance(item, Mapping):
                for key in item.keys():
                    if key in key_set and key not in key_order:
                        key_order.append(key)
                if len(key_order) == len(key_set):
                    break
        if len(key_order) == len(key_set):
            break

    if len(key_order) != len(key_set):
        for key in sorted(key_set, key=lambda x: str(x)):
            if key not in key_order:
                key_order.append(key)
    return key_order


def _validate_key_presence(materialized: Mapping[str, List[Any]], key_order: Sequence[Any]) -> None:
    if not key_order:
        return

    for field, entries in materialized.items():
        for idx, item in enumerate(entries):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"field '{field}' entry at position {idx} is not a mapping but alignment keys were inferred"
                )
            for key in key_order:
                if key not in item:
                    raise ValueError(
                        f"field '{field}' entry at position {idx} is missing alignment key '{key}'"
                    )


def _build_key_rows(
    materialized: Mapping[str, List[Any]],
    key_order: Sequence[Any],
    length: int,
) -> List[Dict[Any, Any]]:
    if length == 0:
        return []
    if not key_order:
        return [{} for _ in range(length)]

    rows: List[Dict[Any, Any]] = []
    for row_index in range(length):
        accumulator: Dict[Any, Any] = {}
        for entries in materialized.values():
            item = entries[row_index]
            if isinstance(item, Mapping):
                for key in key_order:
                    value = item[key]
                    if key in accumulator and accumulator[key] != value:
                        raise ValueError(
                            f"alignment key '{key}' mismatch detected at row {row_index}"
                        )
                    accumulator.setdefault(key, value)
        rows.append({key: accumulator[key] for key in key_order})
    return rows


def _strip_key_columns(
    materialized: Mapping[str, List[Any]],
    key_set: set[Any],
) -> Dict[str, List[Any]]:
    payloads: Dict[str, List[Any]] = {}
    for field, entries in materialized.items():
        payload_list: List[Any] = []
        for item in entries:
            if isinstance(item, Mapping):
                if key_set:
                    payload = {k: item[k] for k in item.keys() if k not in key_set}
                else:
                    payload = dict(item)
                payload_list.append(payload)
            else:
                payload_list.append(item)
        payloads[field] = payload_list
    return payloads


def _select_value(field: str, entry: Any, preferred_key: Any) -> Any:
    if preferred_key is not None:
        if not isinstance(entry, Mapping):
            raise KeyError(
                f"field '{field}' does not provide mapping payloads; cannot select key '{preferred_key}'"
            )
        if preferred_key not in entry:
            available = ", ".join(sorted(str(k) for k in entry.keys()))
            raise KeyError(
                f"field '{field}' payload is missing key '{preferred_key}' (available: {available})"
            )
        return entry[preferred_key]

    if isinstance(entry, Mapping):
        for candidate in _PREFERRED_VALUE_KEYS:
            if candidate in entry:
                return entry[candidate]
        if len(entry) == 1:
            return next(iter(entry.values()))
        return dict(entry)
    return entry


def _copy_entry(entry: Any) -> Any:
    if isinstance(entry, Mapping):
        return dict(entry)
    return entry
