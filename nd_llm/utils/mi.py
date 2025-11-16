"""Mutual-information helper utilities for assembling bottleneck context."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from nd_llm.encoders import Encoder
from nd_llm.metrics import MIProxy


def _normalise_entries(entries: Iterable[Any]) -> list[Any]:
    if isinstance(entries, MutableMapping):
        return [dict(entries)]
    if isinstance(entries, Sequence) and not isinstance(
        entries, (str, bytes, bytearray)
    ):
        result: list[Any] = []
        for item in entries:
            if isinstance(item, MutableMapping):
                result.append(dict(item))
            else:
                result.append(item)
        return result
    return [entries]


def _infer_embedding_dim(
    embeddings: Sequence[Sequence[float]], encoder: Encoder
) -> int:
    dims = [len(vector) for vector in embeddings if len(vector)]
    if dims:
        return int(max(dims))
    return int(getattr(encoder, "embedding_dim", 0) or 0)


def _mean_vector(embeddings: Sequence[Sequence[float]], dim: int) -> list[float]:
    if not embeddings or dim <= 0:
        return []
    totals = [0.0] * dim
    for vector in embeddings:
        for idx in range(dim):
            value = float(vector[idx]) if idx < len(vector) else 0.0
            totals[idx] += value
    scale = 1.0 / float(len(embeddings))
    return [value * scale for value in totals]


def build_mi_proxy_context(
    fields: Mapping[str, Sequence[Any]],
    encoders: Mapping[str, Encoder],
    *,
    preferred_fields: Optional[Sequence[str]] = None,
) -> Tuple[Optional[MIProxy], Optional[Dict[str, Any]]]:
    """Return an :class:`MIProxy` and context mapping for :meth:`IBottleneck.compress`.

    Parameters
    ----------
    fields:
        Mapping of field names to batches supplied to :meth:`IBottleneck.compress`.
    encoders:
        Mapping of field names to encoder instances capable of producing embeddings.
    preferred_fields:
        Optional sequence specifying which fields to prioritise when selecting a
        shared embedding dimensionality for the mutual-information proxy.
    """

    field_order: list[str] = []
    if preferred_fields:
        field_order.extend(str(name) for name in preferred_fields)
    for name in encoders:
        if name not in field_order:
            field_order.append(name)

    proxy: Optional[MIProxy] = None
    active_dim: Optional[int] = None
    targets: Dict[str, list[float]] = {}

    for field in field_order:
        encoder = encoders.get(field)
        if encoder is None:
            continue
        raw_entries = fields.get(field)
        if not raw_entries:
            continue
        batch = _normalise_entries(raw_entries)
        if not batch:
            continue
        try:
            embeddings = encoder.encode(batch)
        except Exception:
            continue
        if not embeddings:
            continue
        dim = _infer_embedding_dim(embeddings, encoder)
        if dim <= 0:
            continue
        if proxy is None:
            try:
                proxy = MIProxy(d_model=dim)
            except Exception:  # pragma: no cover - torch may be unavailable
                return None, None
            active_dim = dim
        if active_dim != dim:
            continue
        targets[field] = _mean_vector(embeddings, dim)

    if proxy is None or not targets:
        return None, None

    target_mapping = {field: list(vector) for field, vector in targets.items()}
    ordered_fields = sorted(target_mapping)
    target_repr = [list(target_mapping[field]) for field in ordered_fields]
    context: Dict[str, Any] = {
        "mi_targets": target_mapping,
        "target_embeddings": target_mapping,
        "target_repr": target_repr,
    }
    return proxy, context


__all__ = ["build_mi_proxy_context"]
