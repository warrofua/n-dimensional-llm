"""Learnable scoring modules for the information bottleneck."""

from __future__ import annotations

from typing import (
    Any,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import torch
from torch import Tensor, nn

DEFAULT_QUERY_KEY = "query_embedding"
DEFAULT_PER_FIELD_QUERY_KEY = "query_embeddings"

if TYPE_CHECKING:  # pragma: no cover
    from .ib import ScoringFn


class LearnableTokenScorer(nn.Module):
    """Simple MLP-based scorer that consumes embeddings and optional context."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        context_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (128,),
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if context_dim is not None and context_dim < 0:
            raise ValueError("context_dim must be non-negative when provided")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")
        self.embedding_dim = int(embedding_dim)
        self.context_dim = int(context_dim) if context_dim is not None else None
        hidden_dims = tuple(int(h) for h in hidden_dims)
        if not hidden_dims:
            hidden_dims = (128,)
        input_dim = self.embedding_dim + (self.context_dim or 0)
        if input_dim <= 0:
            raise ValueError("combined input dimension must be positive")

        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            if hidden <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.append(nn.Linear(in_dim, hidden))
            if activation is None:
                layers.append(nn.ReLU())
            elif isinstance(activation, nn.Module):
                layers.append(activation)
            else:
                layers.append(activation())  # type: ignore[misc]
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: Tensor,
        query: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D tensor [tokens, dim]")
        features = embeddings
        if query is not None:
            expanded_query = self._expand_query(query, embeddings.size(0))
            if self.context_dim is None:
                raise ValueError(
                    "scorer was initialised without context_dim but query was provided"
                )
            if expanded_query.size(1) != self.context_dim:
                raise ValueError(
                    "query/context dimensionality does not match configured context_dim"
                )
        elif self.context_dim is not None:
            expanded_query = embeddings.new_zeros(
                (embeddings.size(0), self.context_dim)
            )
        else:
            expanded_query = None
        if expanded_query is not None:
            features = torch.cat([features, expanded_query], dim=-1)
        return self.network(features).squeeze(-1)

    def _expand_query(self, query: Tensor, target_tokens: int) -> Tensor:
        if query.ndim == 1:
            return query.unsqueeze(0).expand(target_tokens, -1)
        if query.ndim == 2:
            if query.size(0) == 1:
                return query.expand(target_tokens, -1)
            if query.size(0) != target_tokens:
                raise ValueError(
                    "per-token query context must match embedding batch size"
                )
            return query
        raise ValueError("query tensor must be 1D or 2D")


class LearnableScoringStrategy:
    """Adapter that lets :class:`LearnableTokenScorer` plug into :class:`IBottleneck`."""

    def __init__(
        self,
        module: nn.Module,
        *,
        query_key: str = DEFAULT_QUERY_KEY,
        per_field_query_key: str = DEFAULT_PER_FIELD_QUERY_KEY,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be an nn.Module")
        self.module = module
        self.query_key = query_key
        self.per_field_query_key = per_field_query_key

    def __call__(
        self,
        field: str,
        embeddings: Sequence[Sequence[float]],
        metadata: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Tensor:
        param = next(self.module.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        dtype = param.dtype if param is not None else torch.float32
        if not embeddings:
            return torch.zeros(0, device=device, dtype=dtype)
        embedding_tensor = torch.as_tensor(embeddings, device=device, dtype=dtype)
        query_tensor = self._resolve_query_tensor(
            context, field, embedding_tensor, device, dtype
        )
        return self.module(
            embedding_tensor,
            query=query_tensor,
            field=field,
            metadata=metadata,
            context=context,
        )

    def _resolve_query_tensor(
        self,
        context: Mapping[str, Any],
        field: str,
        embeddings: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if not context:
            return None
        per_field = context.get(self.per_field_query_key)
        if isinstance(per_field, Mapping) and field in per_field:
            raw_query = per_field[field]
        else:
            raw_query = context.get(self.query_key)
            if isinstance(raw_query, Mapping):
                raw_query = raw_query.get(field)
        if raw_query is None:
            return None
        query_tensor = torch.as_tensor(raw_query, device=device, dtype=dtype)
        if query_tensor.ndim == 1:
            return query_tensor
        if query_tensor.ndim == 2:
            if query_tensor.size(0) in {1, embeddings.size(0)}:
                return query_tensor
        raise ValueError("query context must be a vector or batch-aligned matrix")


def configure_scorer(
    config: Mapping[str, Any] | str,
) -> Tuple["ScoringFn", Optional[nn.Module]]:
    """Factory helper that normalises scorer configuration."""

    if isinstance(config, str):
        config_map: MutableMapping[str, Any] = {"type": config}
    elif isinstance(config, Mapping):
        config_map = dict(config)
    else:
        raise TypeError("config must be a string or mapping")

    strategy = str(config_map.pop("type", "")).lower()
    if not strategy:
        raise ValueError("config must specify a scorer 'type'")

    if strategy in {"l2", "l2-norm", "norm", "magnitude"}:
        from .ib import NormScoringStrategy

        return NormScoringStrategy(), None
    if strategy in {"query", "query-dot", "query_attention", "attention"}:
        from .ib import QueryDotProductScoringStrategy

        return QueryDotProductScoringStrategy(**config_map), None
    if strategy not in {"learnable", "learned", "trainable"}:
        raise ValueError(f"unknown scorer strategy '{strategy}'")

    embedding_dim = config_map.pop("embedding_dim", None)
    if embedding_dim is None:
        raise ValueError("learnable scorers require 'embedding_dim'")
    hidden_dims = config_map.pop("hidden_dims", (128,))
    context_dim = config_map.pop("context_dim", None)
    dropout = float(config_map.pop("dropout", 0.0))
    activation = config_map.pop("activation", None)
    query_key = config_map.pop("query_key", DEFAULT_QUERY_KEY)
    per_field_query_key = config_map.pop(
        "per_field_query_key", DEFAULT_PER_FIELD_QUERY_KEY
    )
    module = LearnableTokenScorer(
        embedding_dim=int(embedding_dim),
        context_dim=context_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    )
    strategy_fn = LearnableScoringStrategy(
        module,
        query_key=query_key,
        per_field_query_key=per_field_query_key,
    )
    if config_map:
        raise ValueError(
            f"unexpected parameters for learnable scorer: {sorted(config_map)}"
        )
    return strategy_fn, module


__all__ = [
    "LearnableTokenScorer",
    "LearnableScoringStrategy",
    "configure_scorer",
]
