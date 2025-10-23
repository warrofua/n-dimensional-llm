"""High-level encoder/decoder scaffold combining ND-LLM components."""


from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nd_llm.bottleneck import CompressionResult, CompressionTelemetry, IBottleneck
from nd_llm.metrics import MIProxy
from nd_llm.encoders import Encoder
from nd_llm.registry import Registry
from nd_llm.utils import PackedFields

__all__ = ["NDEncoderDecoder"]


def _normalise_identifier(value: Any) -> str:
    """Convert arbitrary identifiers to a stable string representation."""

    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "::".join(_normalise_identifier(item) for item in value)
    return repr(value)


def _materialise_entries(raw_entries: Iterable[Any]) -> List[Any]:
    entries: List[Any] = []
    for item in raw_entries:
        if isinstance(item, MutableMapping):
            entries.append(dict(item))
        else:
            entries.append(item)
    return entries


@dataclass(frozen=True)
class _AggregatedBatch:
    tokens: Tensor
    mask: Tensor
    metadata: List[List[Mapping[str, Any]]]
    doc_order: List[str]
    doc_values: List[Any]
    token_counts: List[int]


class CanonicalCellAggregator:
    """Aggregate registered field tokens into per-document sequences."""

    def __init__(
        self,
        *,
        registry: Registry,
        projections: nn.ModuleDict,
        value_keys: MutableMapping[str, Optional[str]],
        hidden_dim: int,
    ) -> None:
        self._registry = registry
        self._projections = projections
        self._value_keys = value_keys
        self._hidden_dim = int(hidden_dim)

    def aggregate(
        self,
        fields: Mapping[str, Sequence[Any]],
        *,
        doc_ids: Optional[Sequence[Any]] = None,
    ) -> _AggregatedBatch:
        device = self._resolve_device()
        if not fields:
            if doc_ids is None:
                empty_tokens = torch.zeros(0, 0, self._hidden_dim, device=device)
                empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
                return _AggregatedBatch(
                    tokens=empty_tokens,
                    mask=empty_mask,
                    metadata=[],
                    doc_order=[],
                    doc_values=[],
                    token_counts=[],
                )
            doc_order_empty: List[str] = []
            doc_values_empty: List[Any] = []
            for raw in doc_ids:
                normalised = _normalise_identifier(raw)
                doc_order_empty.append(normalised)
                doc_values_empty.append(raw)
            tokens = torch.zeros(len(doc_order_empty), 0, self._hidden_dim, device=device)
            mask = torch.zeros(len(doc_order_empty), 0, dtype=torch.bool, device=device)
            empty_metadata: List[List[Mapping[str, Any]]] = [[] for _ in doc_order_empty]
            empty_token_counts = [0 for _ in doc_order_empty]
            return _AggregatedBatch(
                tokens=tokens,
                mask=mask,
                metadata=empty_metadata,
                doc_order=doc_order_empty,
                doc_values=doc_values_empty,
                token_counts=empty_token_counts,
            )
        per_doc: Dict[str, List[Tuple[Tensor, Mapping[str, Any]]]] = defaultdict(list)
        doc_lookup: Dict[str, Any] = {}

        for field_name in sorted(fields):
            if field_name not in self._registry.fields:
                raise KeyError(f"Field '{field_name}' is not registered with the model")
            entries = _materialise_entries(fields[field_name])
            if not entries:
                continue
            encoder = self._registry.get_encoder(field_name)
            value_key = self._value_keys.get(field_name)
            encoded_inputs = [self._select_value(entry, value_key) for entry in entries]
            embeddings = encoder.encode(encoded_inputs)
            if not embeddings:
                continue
            embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
            try:
                projection = self._projections[field_name]
            except KeyError as exc:
                raise KeyError(f"No projection registered for field '{field_name}'") from exc
            projected = projection(embedding_tensor)
            field_spec = self._registry.fields[field_name]
            keys = list(field_spec.keys)

            for idx, entry in enumerate(entries):
                entry_mapping = entry if isinstance(entry, Mapping) else {}
                key_values = self._resolve_key_tuple(entry_mapping, keys)
                doc_value = key_values[0] if key_values else None
                if doc_value is None:
                    doc_key = _normalise_identifier((field_name, idx))
                else:
                    doc_key = _normalise_identifier(doc_value)
                doc_lookup.setdefault(doc_key, doc_value)
                sort_key = (
                    tuple(_normalise_identifier(value) for value in key_values),
                    field_name,
                    idx,
                )
                coords = self._extract_coords(entry)
                metadata_entry: Dict[str, Any] = {
                    "field": field_name,
                    "keys": tuple(key_values),
                    "doc_id": doc_lookup.get(doc_key),
                    "index": idx,
                    "value": self._select_value(entry, value_key),
                    "sort_key": sort_key,
                }
                if coords is not None:
                    metadata_entry["coords"] = tuple(float(value) for value in coords)
                per_doc[doc_key].append((projected[idx], metadata_entry))

        doc_order: List[str]
        if doc_ids is not None:
            doc_order = []
            for raw in doc_ids:
                normalised = _normalise_identifier(raw)
                doc_order.append(normalised)
                per_doc.setdefault(normalised, [])
                doc_lookup.setdefault(normalised, raw)
        else:
            doc_order = sorted(per_doc)

        if not doc_order:
            empty_tokens = torch.zeros(0, 0, self._hidden_dim, device=device)
            empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
            return _AggregatedBatch(
                tokens=empty_tokens,
                mask=empty_mask,
                metadata=[],
                doc_order=[],
                doc_values=[],
                token_counts=[],
            )

        ordered_tokens: List[Tensor] = []
        ordered_mask: List[Tensor] = []
        ordered_meta: List[List[Mapping[str, Any]]] = []
        token_counts: List[int] = []
        hidden_dim = self._hidden_dim

        max_tokens = max((len(per_doc[key]) for key in doc_order), default=0)
        for doc_key in doc_order:
            tokens = per_doc.get(doc_key, [])
            tokens.sort(key=lambda item: item[1]["sort_key"])  # type: ignore[index]
            embeddings = [item[0] for item in tokens]
            token_meta: List[Mapping[str, Any]] = []
            for _, meta in tokens:
                meta_dict = dict(meta)
                meta_dict.pop("sort_key", None)
                token_meta.append(meta_dict)
            count = len(embeddings)
            token_counts.append(count)
            if embeddings:
                stacked = torch.stack(embeddings, dim=0)
                mask = torch.ones(count, dtype=torch.bool, device=device)
            else:
                stacked = torch.zeros(0, hidden_dim, device=device)
                mask = torch.zeros(0, dtype=torch.bool, device=device)

            if count < max_tokens:
                pad = torch.zeros(max_tokens - count, hidden_dim, device=device)
                if count:
                    stacked = torch.cat([stacked, pad], dim=0)
                else:
                    stacked = pad
                pad_mask = torch.zeros(max_tokens - count, dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad_mask], dim=0)
            ordered_tokens.append(stacked)
            ordered_mask.append(mask)
            ordered_meta.append(token_meta)

        tokens_tensor = torch.stack(ordered_tokens, dim=0)
        mask_tensor = torch.stack(ordered_mask, dim=0)
        doc_values = [doc_lookup.get(key) for key in doc_order]
        return _AggregatedBatch(
            tokens=tokens_tensor,
            mask=mask_tensor,
            metadata=ordered_meta,
            doc_order=doc_order,
            doc_values=doc_values,
            token_counts=token_counts,
        )

    def _select_value(self, entry: Any, value_key: Optional[str]) -> Any:
        if value_key is None:
            return entry
        if isinstance(entry, Mapping) and value_key in entry:
            return entry[value_key]
        return entry

    def _extract_coords(self, entry: Any) -> Optional[Sequence[float]]:
        if isinstance(entry, Mapping):
            if "xyxy" in entry and isinstance(entry["xyxy"], Sequence):
                return [float(value) for value in entry["xyxy"][:4]]
            coords = entry.get("coords")
            if isinstance(coords, Sequence):
                return [float(value) for value in coords[:4]]
            if all(key in entry for key in ("x", "y")):
                return [float(entry.get("x", 0.0)), float(entry.get("y", 0.0))]
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
            return [float(value) for value in entry[:4]]
        return None

    @staticmethod
    def _resolve_key_tuple(entry: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, ...]:
        if not keys:
            return tuple()
        return tuple(entry.get(key) for key in keys)

    def _resolve_device(self) -> torch.device:
        for module in self._projections.values():
            for param in module.parameters():
                return param.device
        return torch.device("cpu")


class TokenBottleneck(nn.Module):
    """Simple top-k token selector with a learnable scoring MLP."""

    def __init__(self, hidden_dim: int, scorer_hidden: int = 256) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, scorer_hidden),
            nn.ReLU(),
            nn.Linear(scorer_hidden, 1),
        )

    def forward(
        self,
        tokens: Tensor,
        *,
        budget: int,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError("tokens tensor must have shape (batch, num_tokens, hidden_dim)")
        batch, num_tokens, hidden_dim = tokens.shape
        if num_tokens == 0:
            empty = tokens.new_zeros(batch, 0, hidden_dim)
            indices = torch.zeros(batch, 0, dtype=torch.long, device=tokens.device)
            scores = tokens.new_full((batch, 0), float("-inf"))
            selected_mask = tokens.new_zeros(batch, 0, dtype=torch.bool)
            return empty, indices, scores, selected_mask

        scores = self.scorer(tokens).squeeze(-1)
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError("mask must match the first two dimensions of tokens")
            scores = scores.masked_fill(~mask, float("-inf"))
        if budget <= 0:
            empty = tokens.new_zeros(batch, 0, hidden_dim)
            indices = torch.zeros(batch, 0, dtype=torch.long, device=tokens.device)
            selected_mask = tokens.new_zeros(batch, 0, dtype=torch.bool)
            return empty, indices, scores, selected_mask

        k = min(int(budget), num_tokens)
        topk = torch.topk(scores, k=k, dim=-1)
        indices = topk.indices
        gathered = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
        if mask is not None:
            selected_mask = torch.gather(mask, 1, indices)
            gathered = torch.where(selected_mask.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        else:
            selected_mask = torch.ones_like(indices, dtype=torch.bool)
        return gathered, indices, scores, selected_mask


class DecoderStub(nn.Module):
    """Lightweight decoder that pools selected tokens and projects to logits."""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if context.ndim != 3:
            raise ValueError("context must have shape (batch, num_tokens, hidden_dim)")
        if context.size(1) == 0:
            pooled = context.new_zeros(context.size(0), context.size(2))
        elif mask is not None and mask.size(1) == context.size(1):
            weights = mask.to(context.dtype).unsqueeze(-1)
            totals = weights.sum(dim=1).clamp_min(1.0)
            pooled = (context * weights).sum(dim=1) / totals
        else:
            pooled = context.mean(dim=1)
        return self.mlp(pooled)


class NDEncoderDecoder(nn.Module):
    """End-to-end scaffold wiring registry, aggregation, bottleneck, and decoder."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_classes: int = 2,
        bottleneck: Optional[IBottleneck] = None,
        decoder: Optional[nn.Module] = None,
        mi_proxy: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.registry = Registry()
        self._projections: nn.ModuleDict = nn.ModuleDict()
        self._value_keys: Dict[str, Optional[str]] = {}
        self.aggregator = CanonicalCellAggregator(
            registry=self.registry,
            projections=self._projections,
            value_keys=self._value_keys,
            hidden_dim=self.hidden_dim,
        )
        if bottleneck is None:
            bottleneck = IBottleneck(target_budget=1)
        self.bottleneck = bottleneck
        self.decoder = decoder or DecoderStub(self.hidden_dim, self.num_classes)
        self.mi = mi_proxy or MIProxy(d_model=self.hidden_dim)
        self._target_projector = nn.Sequential(
            nn.Linear(self.num_classes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def register_field(
        self,
        name: str,
        *,
        encoder: Encoder,
        keys: Sequence[str],
        salience: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        value_key: Optional[str] = None,
    ) -> None:
        """Register a field specification, encoder, and projection."""

        extra_metadata = dict(metadata or {})
        if name not in self.registry.fields:
            self.registry.add_field(name, keys=list(keys), salience=salience, **extra_metadata)
        else:
            spec = self.registry.fields[name]
            if list(keys) != list(spec.keys):
                raise ValueError(
                    f"Field '{name}' already registered with keys {spec.keys}; received {list(keys)}"
                )
        self.registry.register_encoder(name, encoder)
        projection = nn.Sequential(
            nn.Linear(encoder.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self._projections[name] = projection
        self._value_keys[name] = value_key

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        token_budget: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        context_mapping = dict(context or {})
        raw_fields = batch.get("fields", batch)
        field_mapping = self._normalise_fields(raw_fields)
        doc_ids = batch.get("doc_ids")
        aggregated = self.aggregator.aggregate(field_mapping, doc_ids=doc_ids)
        tokens = aggregated.tokens
        mask = aggregated.mask
        doc_order = aggregated.doc_order
        doc_values = aggregated.doc_values
        metadata = aggregated.metadata

        targets_tensor: Optional[Tensor] = None
        reorder_indices: Optional[Tensor] = None
        targets_input = batch.get("targets")
        if targets_input is not None:
            targets_tensor = self._ensure_tensor(targets_input, dtype=torch.long, device=tokens.device)
            targets_tensor = targets_tensor.view(-1)
            targets_tensor, reorder_indices = self._reorder_targets(
                targets_tensor,
                doc_ids,
                doc_order,
            )

        target_repr_input = batch.get("target_repr")
        if target_repr_input is None:
            target_repr_input = batch.get("target_embeddings")
        if target_repr_input is not None:
            target_repr = self._ensure_tensor(
                target_repr_input,
                dtype=torch.float32,
                device=tokens.device,
            )
            if reorder_indices is not None and target_repr.size(0) == reorder_indices.size(0):
                target_repr = target_repr.index_select(0, reorder_indices)
        elif targets_tensor is not None:
            target_repr = self._build_target_repr(targets_tensor)
        else:
            target_repr = None

        compression_result: Optional[CompressionResult] = None
        telemetry: Optional[CompressionTelemetry]
        compression_metrics: Dict[str, float] = {}
        compression_loss_terms: Dict[str, Any] = {}
        budget_value = int(token_budget)
        has_fields = bool(field_mapping)
        if has_fields and budget_value > 0:
            self.bottleneck.target_budget = max(1, budget_value)
            compression_result = self.bottleneck.compress(
                field_mapping,
                self.registry.encoders,
                registry=self.registry,
                field_specs=self.registry.fields,
                context=context_mapping,
            )
            telemetry = compression_result.telemetry
            compression_metrics = {str(k): float(v) for k, v in compression_result.metrics.items()}
            compression_loss_terms = dict(compression_result.loss_terms)
        else:
            telemetry = None

        (
            selected_tokens,
            selected_mask,
            selected_metadata,
            token_indices,
            token_scores,
        ) = self._gather_selected_tokens(tokens, mask, metadata, telemetry)

        logits = self.decoder(selected_tokens, selected_mask if selected_mask.numel() else None)

        batch_size = selected_tokens.size(0)
        hidden_dim = selected_tokens.size(2) if selected_tokens.ndim == 3 else self.hidden_dim
        if batch_size == 0:
            pooled_tokens = selected_tokens.new_zeros((0, hidden_dim))
        elif selected_tokens.size(1) == 0 or not selected_mask.numel():
            pooled_tokens = selected_tokens.new_zeros((batch_size, hidden_dim))
        else:
            mask_f = selected_mask.unsqueeze(-1).float()
            masked_tokens = selected_tokens * mask_f
            token_totals = masked_tokens.sum(dim=1)
            mask_totals = mask_f.sum(dim=1)
            pooled_tokens = torch.where(
                mask_totals > 0,
                token_totals / mask_totals.clamp_min(1.0),
                torch.zeros_like(token_totals),
            )

        if selected_mask.numel():
            tokens_selected_counts = selected_mask.sum(dim=1)
            has_any_tokens = bool(tokens_selected_counts.sum().item())
        else:
            tokens_selected_counts = torch.zeros(batch_size, device=selected_tokens.device)
            has_any_tokens = False

        if target_repr is not None and has_any_tokens:
            mi_lb_tensor, _ = self.mi(pooled_tokens, target_repr)
        else:
            mi_lb_tensor = pooled_tokens.new_zeros((), dtype=pooled_tokens.dtype, requires_grad=True)

        tokens_selected = tokens_selected_counts
        tokens_available = (
            mask.sum(dim=1) if mask.numel() else torch.zeros(tokens.size(0), device=tokens.device)
        )

        logs: Dict[str, Any] = {
            "mi_lb": float(mi_lb_tensor.detach().cpu()),
            "mi_lb_tensor": mi_lb_tensor,
            "tokens_selected": tokens_selected.detach(),
            "tokens_available": tokens_available.detach(),
            "token_indices": token_indices.detach(),
            "token_scores": token_scores.detach(),
            "token_mask": selected_mask.detach(),
            "doc_ids": doc_values,
            "doc_order": doc_order,
            "cell_mask": mask.detach(),
            "cell_metadata": metadata,
            "selected_metadata": selected_metadata,
            "compression_metrics": compression_metrics,
            "compression_telemetry": telemetry,
            "compression_loss_terms": compression_loss_terms,
        }
        if compression_result is not None:
            logs["compressed_fields"] = compression_result.compressed_fields
        if targets_tensor is not None:
            logs["targets"] = targets_tensor.detach()
        if target_repr is not None:
            logs["target_repr"] = target_repr.detach()
        return logits, logs

    def _gather_selected_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        metadata: Sequence[Sequence[Mapping[str, Any]]],
        telemetry: Optional[CompressionTelemetry],
    ) -> Tuple[Tensor, Tensor, List[List[Mapping[str, Any]]], Tensor, Tensor]:
        batch_size = tokens.size(0)
        device = tokens.device
        hidden_dim = tokens.size(2) if tokens.ndim == 3 else self.hidden_dim
        if batch_size == 0:
            empty_tokens = tokens.new_zeros((0, 0, hidden_dim))
            empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
            empty_indices = torch.zeros(0, 0, dtype=torch.long, device=device)
            empty_scores = tokens.new_zeros((0, 0))
            return empty_tokens, empty_mask, [], empty_indices, empty_scores

        if telemetry is None or not any(len(v) for v in telemetry.selected_indices.values()):
            empty_tokens = tokens.new_zeros((batch_size, 0, hidden_dim))
            empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            empty_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            empty_scores = tokens.new_zeros((batch_size, 0))
            empty_metadata: List[List[Mapping[str, Any]]] = [[] for _ in range(batch_size)]
            return empty_tokens, empty_mask, empty_metadata, empty_indices, empty_scores

        selected_lookup: Dict[str, set[int]] = {}
        for field, indices in telemetry.selected_indices.items():
            selected_lookup[str(field)] = {int(idx) for idx in indices}

        score_lookup: Dict[str, Dict[int, float]] = {}
        for field, scores in telemetry.selected_scores.items():
            key = str(field)
            field_indices = telemetry.selected_indices.get(field, [])
            score_lookup[key] = {
                int(idx): float(score) for idx, score in zip(field_indices, scores)
            }

        doc_selected_embeddings: List[List[Tensor]] = []
        doc_selected_metadata: List[List[Mapping[str, Any]]] = []
        doc_selected_indices: List[List[int]] = []
        doc_selected_scores: List[List[float]] = []

        for doc_idx in range(batch_size):
            doc_meta_seq = list(metadata[doc_idx]) if doc_idx < len(metadata) else []
            row_embeddings: List[Tensor] = []
            row_metadata: List[Mapping[str, Any]] = []
            row_indices: List[int] = []
            row_scores: List[float] = []
            mask_row: Optional[Tensor]
            if mask.size(0) > doc_idx:
                mask_row = mask[doc_idx]
            else:
                mask_row = None
            for pos, entry in enumerate(doc_meta_seq):
                if mask_row is not None and (mask_row.size(0) <= pos or not mask_row[pos]):
                    continue
                field = str(entry.get("field"))
                if field not in selected_lookup:
                    continue
                index_raw = entry.get("index")
                if index_raw is None:
                    continue
                try:
                    index_val = int(index_raw)
                except (TypeError, ValueError):
                    continue
                if index_val not in selected_lookup[field]:
                    continue
                row_embeddings.append(tokens[doc_idx, pos])
                row_metadata.append(dict(entry))
                row_indices.append(index_val)
                score = score_lookup.get(field, {}).get(index_val, float("-inf"))
                row_scores.append(float(score))
            doc_selected_embeddings.append(row_embeddings)
            doc_selected_metadata.append(row_metadata)
            doc_selected_indices.append(row_indices)
            doc_selected_scores.append(row_scores)

        max_selected = max((len(items) for items in doc_selected_embeddings), default=0)
        selected_metadata = doc_selected_metadata

        if max_selected == 0:
            empty_tokens = tokens.new_zeros((batch_size, 0, hidden_dim))
            empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            empty_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            empty_scores = tokens.new_zeros((batch_size, 0))
            return empty_tokens, empty_mask, selected_metadata, empty_indices, empty_scores

        selected_tokens = tokens.new_zeros((batch_size, max_selected, hidden_dim))
        selected_mask = torch.zeros(batch_size, max_selected, dtype=torch.bool, device=device)
        index_tensor = torch.full((batch_size, max_selected), -1, dtype=torch.long, device=device)
        score_tensor = tokens.new_full((batch_size, max_selected), float("-inf"))

        for doc_idx in range(batch_size):
            embeddings = doc_selected_embeddings[doc_idx]
            count = len(embeddings)
            if count == 0:
                continue
            stacked = torch.stack(embeddings, dim=0)
            selected_tokens[doc_idx, :count] = stacked
            selected_mask[doc_idx, :count] = True
            index_tensor[doc_idx, :count] = torch.tensor(
                doc_selected_indices[doc_idx], dtype=torch.long, device=device
            )
            score_tensor[doc_idx, :count] = torch.tensor(
                doc_selected_scores[doc_idx], dtype=tokens.dtype, device=device
            )

        return selected_tokens, selected_mask, selected_metadata, index_tensor, score_tensor

    def _normalise_fields(
        self,
        fields: Mapping[str, Sequence[Any]] | PackedFields,
    ) -> Mapping[str, Sequence[Any]]:
        if isinstance(fields, PackedFields):
            return fields.with_keys()
        if isinstance(fields, Mapping):
            return {name: list(entries) for name, entries in fields.items()}
        raise TypeError("fields must be a mapping or PackedFields instance")

    def _ensure_tensor(self, value: Any, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        tensor = torch.as_tensor(value, dtype=dtype)
        return tensor.to(device=device)

    def _reorder_targets(
        self,
        targets: Tensor,
        doc_ids: Optional[Sequence[Any]],
        doc_order: Sequence[str],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not doc_order:
            return targets, None
        if doc_ids is None:
            limit = min(len(doc_order), targets.size(0))
            index_tensor = torch.arange(limit, device=targets.device)
            return targets.index_select(0, index_tensor), index_tensor
        mapping = {
            _normalise_identifier(doc_id): idx for idx, doc_id in enumerate(doc_ids)
        }
        order_indices: List[int] = []
        for key in doc_order:
            if key not in mapping:
                raise KeyError(f"Document id {key!r} missing from provided doc_ids")
            order_indices.append(mapping[key])
        index_tensor = torch.tensor(order_indices, dtype=torch.long, device=targets.device)
        return targets.index_select(0, index_tensor), index_tensor

    def _build_target_repr(self, targets: Tensor) -> Tensor:
        if targets.numel() == 0:
            return targets.new_zeros((0, self.hidden_dim), dtype=torch.float32)
        primary = targets.view(targets.size(0), -1)[:, 0]
        valid = primary != -100
        primary = torch.where(valid, primary, torch.zeros_like(primary))
        one_hot = F.one_hot(primary.to(torch.long), num_classes=self.num_classes).to(
            dtype=torch.float32, device=targets.device
        )
        one_hot = one_hot * valid.unsqueeze(-1)
        return self._target_projector(one_hot)
