"""High-level encoder/decoder scaffold combining ND-LLM components."""


from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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
        if not fields:
            raise ValueError("fields mapping must not be empty")
        device = self._resolve_device()
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
                metadata: Dict[str, Any] = {
                    "field": field_name,
                    "keys": tuple(key_values),
                    "doc_id": doc_lookup.get(doc_key),
                    "index": idx,
                    "value": self._select_value(entry, value_key),
                    "sort_key": sort_key,
                }
                per_doc[doc_key].append((projected[idx], metadata))

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


class MIProxy(nn.Module):
    """InfoNCE-style mutual-information proxy."""

    def __init__(self, hidden_dim: int, projection_dim: int = 256, temperature: float = 0.07) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.target = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.temperature = float(temperature)

    def forward(self, tokens: Tensor, target_repr: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if target_repr is None or tokens.size(0) == 0:
            logits = tokens.new_zeros((tokens.size(0), 0))
            return tokens.new_tensor(0.0), logits
        if target_repr.size(0) == 0:
            logits = tokens.new_zeros((tokens.size(0), 0))
            return tokens.new_tensor(0.0), logits
        pooled = tokens.mean(dim=1)
        query = F.normalize(self.query(pooled), dim=-1)
        target = F.normalize(self.target(target_repr), dim=-1)
        logits = (query @ target.T) / self.temperature
        if logits.size(0) == 0:
            return tokens.new_tensor(0.0), logits
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return -loss, logits


class NDEncoderDecoder(nn.Module):
    """End-to-end scaffold wiring registry, aggregation, bottleneck, and decoder."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_classes: int = 2,
        bottleneck: Optional[TokenBottleneck] = None,
        decoder: Optional[nn.Module] = None,
        mi_proxy: Optional[MIProxy] = None,
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
        self.bottleneck = bottleneck or TokenBottleneck(self.hidden_dim)
        self.decoder = decoder or DecoderStub(self.hidden_dim, self.num_classes)
        self.mi = mi_proxy or MIProxy(self.hidden_dim)
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
        del context
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

        selected_tokens, indices, scores, selected_mask = self.bottleneck(
            tokens, budget=token_budget, mask=mask
        )
        if indices.numel():
            selected_scores = torch.gather(scores, 1, indices)
            if selected_mask is not None:
                selected_scores = torch.where(
                    selected_mask, selected_scores, torch.full_like(selected_scores, float("-inf"))
                )
        else:
            selected_scores = scores.new_zeros(scores.size(0), 0)

        logits = self.decoder(selected_tokens, selected_mask)

        mi_lb_tensor, _ = self.mi(selected_tokens, target_repr)
        tokens_selected = selected_mask.sum(dim=1) if selected_mask.numel() else torch.zeros(
            tokens.size(0), device=tokens.device
        )
        tokens_available = mask.sum(dim=1) if mask.numel() else torch.zeros(
            tokens.size(0), device=tokens.device
        )

        selected_metadata: List[List[Mapping[str, Any]]] = []
        for doc_idx, doc_indices in enumerate(indices.tolist() if indices.numel() else []):
            doc_meta = metadata[doc_idx] if doc_idx < len(metadata) else []
            doc_selected: List[Mapping[str, Any]] = []
            for pos, cell_index in enumerate(doc_indices):
                if selected_mask[doc_idx, pos] if selected_mask.numel() else False:
                    if cell_index < len(doc_meta):
                        doc_selected.append(doc_meta[cell_index])
            selected_metadata.append(doc_selected)
        if not selected_metadata and indices.size(0):
            selected_metadata = [[] for _ in range(indices.size(0))]

        logs: Dict[str, Any] = {
            "mi_lb": float(mi_lb_tensor.detach().cpu()),
            "mi_lb_tensor": mi_lb_tensor,
            "tokens_selected": tokens_selected.detach(),
            "tokens_available": tokens_available.detach(),
            "token_indices": indices.detach(),
            "token_scores": selected_scores.detach(),
            "token_mask": selected_mask.detach(),
            "doc_ids": doc_values,
            "doc_order": doc_order,
            "cell_mask": mask.detach(),
            "cell_metadata": metadata,
            "selected_metadata": selected_metadata,
        }
        if targets_tensor is not None:
            logs["targets"] = targets_tensor.detach()
        if target_repr is not None:
            logs["target_repr"] = target_repr.detach()
        return logits, logs

    def _normalise_fields(
        self,
        fields: Mapping[str, Sequence[Any]] | PackedFields,
    ) -> Mapping[str, Sequence[Any]]:
        if isinstance(fields, PackedFields):
            return fields.as_dict()
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
