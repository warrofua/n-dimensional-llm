"""Information bottleneck with pluggable scoring and telemetry."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from nd_llm.encoders import Encoder
from .learnable import LearnableScoringStrategy, configure_scorer

try:  # pragma: no cover - torch is an optional heavy dependency during import
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from nd_llm.registry import FieldSpec, Registry
    from nd_llm.metrics import MIProxy

ScoreVector = List[float]
Embedding = Sequence[float]
EmbeddingBatch = Sequence[Embedding]
FieldMetadata = Mapping[str, Any]
ScoreOutput = Union[ScoreVector, "Tensor"]
ScoringFn = Callable[[str, EmbeddingBatch, FieldMetadata, Mapping[str, Any]], ScoreOutput]
BudgetAllocation = Tuple[Dict[str, int], Dict[str, float]]
BudgetAllocatorFn = Callable[[Mapping[str, Sequence[float]], Mapping[str, FieldMetadata], int], BudgetAllocation]


@dataclass
class CompressionTelemetry:
    """Telemetry collected during compression for later analytics."""

    selected_indices: Dict[str, List[int]]
    selected_scores: Dict[str, List[float]]
    token_counts: Dict[str, int]
    budget: int
    field_budgets: Dict[str, int]
    allocation_weights: Dict[str, float]
    dropped_indices: Dict[str, List[int]]
    residual_statistics: Dict[str, Dict[str, float]]
    quantized_embeddings: Dict[str, List[Dict[str, Any]]]


@dataclass
class CompressionResult:
    """Container returned by :meth:`IBottleneck.compress`."""

    compressed_fields: Dict[str, List[Any]]
    telemetry: CompressionTelemetry
    metrics: Dict[str, float]
    loss_terms: Dict[str, Any] = field(default_factory=dict)


class NormScoringStrategy:
    """Score tokens by their L2 norm."""

    def __call__(
        self,
        field: str,
        embeddings: EmbeddingBatch,
        metadata: FieldMetadata,
        context: Mapping[str, Any],
    ) -> ScoreVector:
        return [_vector_norm(vector) for vector in embeddings]


class QueryDotProductScoringStrategy:
    """Blend per-token norms with query-conditioned attention scores."""

    def __init__(
        self,
        *,
        fallback: Optional[ScoringFn] = None,
        normalize: bool = True,
        mix_weight: float = 0.5,
        query_key: str = "query_embedding",
        per_field_query_key: str = "query_embeddings",
    ) -> None:
        if mix_weight < 0.0 or mix_weight > 1.0:
            raise ValueError("mix_weight must fall within [0, 1]")
        self.fallback: ScoringFn
        if fallback is None:
            self.fallback = cast(ScoringFn, NormScoringStrategy())
        else:
            self.fallback = fallback
        self.normalize = bool(normalize)
        self.mix_weight = mix_weight
        self.query_key = query_key
        self.per_field_query_key = per_field_query_key

    def __call__(
        self,
        field: str,
        embeddings: EmbeddingBatch,
        metadata: FieldMetadata,
        context: Mapping[str, Any],
    ) -> ScoreVector:
        base_scores = self.fallback(field, embeddings, metadata, context)
        query = self._resolve_query(context, field)
        if query is None:
            return base_scores

        query_vector = [float(component) for component in query]
        query_norm = _vector_norm(query_vector)
        if query_norm == 0.0:
            return base_scores

        scores: ScoreVector = []
        for base_score, embedding in zip(base_scores, embeddings):
            if not embedding:
                scores.append(0.0)
                continue
            dot_product = sum(float(a) * float(b) for a, b in zip(embedding, query_vector))
            if self.normalize:
                denom = query_norm * (base_score if base_score != 0.0 else _vector_norm(embedding))
                attention_score = dot_product / denom if denom else 0.0
            else:
                attention_score = dot_product
            if self.mix_weight == 1.0:
                combined = attention_score
            elif self.mix_weight == 0.0:
                combined = base_score
            else:
                combined = self.mix_weight * attention_score + (1.0 - self.mix_weight) * base_score
            scores.append(combined)
        return scores

    def _resolve_query(self, context: Mapping[str, Any], field: str) -> Optional[Sequence[float]]:
        if not context:
            return None
        per_field = context.get(self.per_field_query_key)
        if isinstance(per_field, Mapping) and field in per_field:
            return per_field[field]
        generic = context.get(self.query_key)
        if isinstance(generic, Mapping):
            if field in generic:
                return generic[field]
        elif generic is not None:
            return generic
        return None


class RegistryAwareBudgetAllocator:
    """Allocate field-level budgets using registry metadata."""

    def __init__(
        self,
        *,
        salience_bonus: float = 1.5,
        key_weight: float = 0.3,
        min_weight: float = 0.05,
    ) -> None:
        if salience_bonus <= 0:
            raise ValueError("salience_bonus must be positive")
        if key_weight < 0:
            raise ValueError("key_weight must be non-negative")
        if min_weight <= 0:
            raise ValueError("min_weight must be positive")
        self.salience_bonus = salience_bonus
        self.key_weight = key_weight
        self.min_weight = min_weight

    def __call__(
        self,
        scores: Mapping[str, Sequence[float]],
        metadata: Mapping[str, FieldMetadata],
        total_budget: int,
    ) -> BudgetAllocation:
        token_counts = {field: len(values) for field, values in scores.items()}
        available = sum(token_counts.values())
        limit = min(int(total_budget), available)
        weights: Dict[str, float] = {}
        for field in scores:
            info = metadata.get(field, {})
            budget_weight = info.get("budget_weight") if isinstance(info, Mapping) else None
            if isinstance(budget_weight, (int, float)):
                weight = float(budget_weight)
            else:
                keys = info.get("keys", []) if isinstance(info, Mapping) else []
                salience = bool(info.get("salience")) if isinstance(info, Mapping) else False
                weight = 1.0 + self.key_weight * len(keys)
                if salience:
                    weight *= self.salience_bonus
            weights[field] = float(weight if weight > 0 else self.min_weight)

        eligible_fields = [field for field, count in token_counts.items() if count > 0]
        if limit <= 0 or not eligible_fields:
            return ({field: 0 for field in scores}, {field: weights.get(field, 0.0) for field in scores})

        total_weight = sum(weights[field] for field in eligible_fields)
        if total_weight <= 0:
            total_weight = float(len(eligible_fields))

        allocations: Dict[str, int] = {field: 0 for field in scores}
        remainders: List[Tuple[float, float, int, str]] = []
        assigned = 0
        for field in scores:
            count = token_counts[field]
            if count == 0:
                remainders.append((0.0, weights[field], count, field))
                continue
            share = limit * (weights[field] / total_weight)
            share = min(share, float(count))
            base = int(math.floor(share))
            allocations[field] = base
            assigned += base
            remainder = share - base
            remainders.append((remainder, weights[field], count, field))

        salience_fields = [
            field
            for field, info in metadata.items()
            if token_counts.get(field, 0) > 0 and isinstance(info, Mapping) and bool(info.get("salience"))
        ]
        for field in salience_fields:
            if assigned >= limit:
                break
            capacity = token_counts[field]
            if allocations[field] < capacity:
                allocations[field] += 1
                assigned += 1

        remainders.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        idx = 0
        while assigned < limit and idx < len(remainders):
            _, _, capacity, field = remainders[idx]
            idx += 1
            if capacity == 0:
                continue
            if allocations[field] < capacity:
                allocations[field] += 1
                assigned += 1

        if assigned < limit:
            for field, capacity in token_counts.items():
                while assigned < limit and allocations[field] < capacity:
                    allocations[field] += 1
                    assigned += 1

        return allocations, {field: weights.get(field, 0.0) for field in scores}


class IBottleneck:
    """Variable-rate information bottleneck with registry-aware telemetry."""

    def __init__(
        self,
        target_budget: int,
        *,
        objective: Optional[str] = None,
        scorer: Optional[ScoringFn] = None,
        scorer_config: Optional[Mapping[str, Any]] = None,
        learnable_scorer: Optional[nn.Module] = None,
        budget_allocator: Optional[BudgetAllocatorFn] = None,
    ) -> None:
        if target_budget <= 0:
            raise ValueError("target_budget must be positive")
        self.target_budget = int(target_budget)
        self.objective = (objective or "l2-norm").lower()
        configured_module: Optional[nn.Module] = None
        if scorer_config is not None:
            if scorer is not None:
                raise ValueError("specify either 'scorer' or 'scorer_config', not both")
            scorer, configured_module = configure_scorer(scorer_config)
        if learnable_scorer is None and configured_module is not None:
            learnable_scorer = configured_module
        if learnable_scorer is not None and torch is None:  # pragma: no cover - defensive
            raise RuntimeError("torch is required when supplying a learnable scorer")
        if learnable_scorer is not None and scorer is None:
            scorer = LearnableScoringStrategy(learnable_scorer)
        self.learnable_scorer: Optional[nn.Module] = learnable_scorer
        self.scorer = scorer or self._resolve_objective(self.objective)
        self.budget_allocator = budget_allocator or RegistryAwareBudgetAllocator()
        self._score_tensors: Dict[str, Tensor] = {}

    def compress(
        self,
        fields: Mapping[str, Sequence[Any]],
        encoders: Mapping[str, Encoder],
        *,
        registry: Optional["Registry"] = None,
        field_specs: Optional[Mapping[str, "FieldSpec | Mapping[str, Any]"]] = None,
        context: Optional[Mapping[str, Any]] = None,
        mi_proxy: Optional["MIProxy"] = None,
    ) -> CompressionResult:
        if not fields:
            raise ValueError("fields must not be empty")
        metadata = self._normalise_field_metadata(fields.keys(), field_specs, registry)

        encoded = self._encode_fields(fields, encoders)
        scoring_context = context or {}
        self._score_tensors.clear()
        gating_scores = self._compute_scores(encoded, metadata, scoring_context)

        field_budgets, allocation_weights = self.budget_allocator(gating_scores, metadata, self.target_budget)
        selected = self._select_indices(gating_scores, field_budgets)

        compressed_fields: Dict[str, List[Any]] = {}
        token_counts: Dict[str, int] = {}
        selected_scores: Dict[str, List[float]] = {}
        dropped_indices: Dict[str, List[int]] = {}
        residual_statistics: Dict[str, Dict[str, float]] = {}
        quantized_embeddings: Dict[str, List[Dict[str, Any]]] = {}
        for field, indices in selected.items():
            source = list(fields[field])
            compressed_fields[field] = [source[i] for i in indices]
            token_counts[field] = len(source)
            field_scores = gating_scores[field]
            selected_scores[field] = [field_scores[i] for i in indices]
            selected_set = set(indices)
            dropped = sorted(i for i in range(len(field_scores)) if i not in selected_set)
            dropped_indices[field] = dropped
            field_embeddings = encoded.get(field, [])
            kept_embeddings = [field_embeddings[i] for i in indices if i < len(field_embeddings)]
            dropped_embeddings = [field_embeddings[i] for i in dropped if i < len(field_embeddings)]
            residual_statistics[field] = _compute_residual_statistics(
                kept_embeddings,
                dropped_embeddings,
                field_scores,
                indices,
                dropped,
            )
            quantized_embeddings[field] = [
                {
                    "index": int(idx),
                    **_quantize_embedding(field_embeddings[idx] if idx < len(field_embeddings) else []),
                }
                for idx in dropped
            ]

        telemetry = CompressionTelemetry(
            selected_indices={k: list(v) for k, v in selected.items()},
            selected_scores=selected_scores,
            token_counts=token_counts,
            budget=self.target_budget,
            field_budgets=field_budgets,
            allocation_weights=allocation_weights,
            dropped_indices=dropped_indices,
            residual_statistics=residual_statistics,
            quantized_embeddings=quantized_embeddings,
        )
        metrics = self._compute_metrics(encoded, telemetry)
        mi_targets: Optional[Mapping[str, Any]]
        if isinstance(context, Mapping):
            targets = context.get("mi_targets")
            mi_targets = targets if isinstance(targets, Mapping) else None
        else:
            mi_targets = None
        mi_lower_bound = self._compute_mutual_information_lower_bound(
            encoded, telemetry, mi_proxy, mi_targets
        )
        if mi_lower_bound is not None:
            metrics["mi_lower_bound"] = mi_lower_bound
        loss_terms = self._compute_loss_terms(encoded)
        return CompressionResult(
            compressed_fields=compressed_fields,
            telemetry=telemetry,
            metrics=metrics,
            loss_terms=loss_terms,
        )

    def decompress(self, result: CompressionResult) -> Dict[str, Any]:
        """Approximate reconstruction using stored telemetry artefacts."""

        telemetry = result.telemetry
        kept = {field: list(values) for field, values in result.compressed_fields.items()}

        regenerated: Dict[str, List[Dict[str, Any]]] = {}
        field_metrics: Dict[str, Dict[str, float]] = {}
        retained_counts: List[int] = []
        regenerated_counts: List[int] = []
        mse_accumulator = 0.0
        mse_weight = 0
        kl_accumulator = 0.0
        kl_fields = 0

        fields = set(telemetry.token_counts) | set(kept)

        for field in fields:
            total_count = int(telemetry.token_counts.get(field, len(kept.get(field, []))))
            kept_tokens = kept.get(field, [])
            retained_counts.append(len(kept_tokens))

            quantized = telemetry.quantized_embeddings.get(field, [])
            residual = telemetry.residual_statistics.get(field, {})

            reconstructed_embeddings: List[Dict[str, Any]] = []
            for entry in sorted(quantized, key=lambda item: item.get("index", 0)):
                embedding = _dequantize_embedding(entry)
                reconstructed_embeddings.append(
                    {
                        "index": int(entry.get("index", len(reconstructed_embeddings))),
                        "embedding": embedding,
                    }
                )
            regenerated[field] = reconstructed_embeddings
            regenerated_counts.append(len(reconstructed_embeddings))

            dropped_count = int(residual.get("dropped_count", len(reconstructed_embeddings)))
            field_mse = float(residual.get("mean_squared_error", 0.0))
            field_kl = float(residual.get("kl_divergence", 0.0))

            if dropped_count > 0:
                mse_accumulator += field_mse * dropped_count
                mse_weight += dropped_count
            if field_kl:
                kl_accumulator += field_kl
                kl_fields += 1

            field_metrics[field] = {
                "mse": field_mse,
                "kl_divergence": field_kl,
                "kept": float(len(kept_tokens)),
                "regenerated": float(len(reconstructed_embeddings)),
                "total": float(total_count),
            }

        total_kept = sum(retained_counts)
        total_regenerated = sum(regenerated_counts)
        total_tokens = total_kept + total_regenerated
        retained_ratio = float(total_kept) / float(total_tokens) if total_tokens else 1.0
        regenerated_ratio = float(total_regenerated) / float(total_tokens) if total_tokens else 0.0
        mean_mse = mse_accumulator / float(mse_weight or 1)
        mean_kl = kl_accumulator / float(kl_fields or 1)

        metrics = {
            "retained_ratio": retained_ratio,
            "regenerated_ratio": regenerated_ratio,
            "mean_mse": mean_mse,
            "mean_kl_divergence": mean_kl,
            "total_tokens": float(total_tokens),
            "fields": field_metrics,
        }

        return {
            "kept": kept,
            "regenerated": regenerated,
            "metrics": metrics,
        }

    def _encode_fields(
        self,
        fields: Mapping[str, Sequence[Any]],
        encoders: Mapping[str, Encoder],
    ) -> Dict[str, List[List[float]]]:
        encoded: Dict[str, List[List[float]]] = {}
        for field, batch in fields.items():
            if field not in encoders:
                raise KeyError(f"missing encoder for field '{field}'")
            embedding_batch = encoders[field].encode(batch)
            converted = [list(vector) for vector in embedding_batch]
            if len(converted) != len(batch):
                raise ValueError(
                    f"encoder for field '{field}' must return the same number of embeddings as inputs"
                )
            encoded[field] = converted
        return encoded

    def _compute_scores(
        self,
        encoded: Mapping[str, List[List[float]]],
        metadata: Mapping[str, FieldMetadata],
        context: Mapping[str, Any],
    ) -> Dict[str, List[float]]:
        scores: Dict[str, List[float]] = {}
        for field, embeddings in encoded.items():
            raw_scores = self.scorer(field, embeddings, metadata.get(field, {}), context)
            tensor: Optional[Tensor] = None
            if torch is not None and isinstance(raw_scores, torch.Tensor):
                tensor = raw_scores.squeeze(-1) if raw_scores.ndim > 1 else raw_scores
                if tensor.ndim != 1:
                    raise ValueError(
                        "learnable scorer must return a 1D tensor of per-token scores"
                    )
                field_scores = [float(v) for v in tensor.detach().cpu().tolist()]
            else:
                field_scores = [
                    float(v) for v in cast(Sequence[float], raw_scores)
                ]
            if len(field_scores) != len(embeddings):
                raise ValueError(
                    f"scoring strategy returned {len(field_scores)} scores for {len(embeddings)} embeddings in field '{field}'"
                )
            if tensor is not None:
                self._score_tensors[field] = tensor
            scores[field] = field_scores
        return scores

    def _select_indices(
        self,
        scores: Mapping[str, List[float]],
        budgets: Mapping[str, int],
    ) -> Dict[str, List[int]]:
        flat: List[tuple[float, str, int]] = []
        for field, values in scores.items():
            for idx, score in enumerate(values):
                flat.append((float(score), field, idx))
        flat.sort(key=lambda item: (-item[0], item[1], item[2]))

        total_budget = sum(max(0, int(budgets.get(field, 0))) for field in scores)
        if total_budget == 0:
            return {field: [] for field in scores}

        selected: Dict[str, List[int]] = {field: [] for field in scores}
        counts: Dict[str, int] = {field: 0 for field in scores}
        taken = 0
        for score, field, idx in flat:
            budget = max(0, int(budgets.get(field, 0)))
            if counts[field] >= budget:
                continue
            selected[field].append(idx)
            counts[field] += 1
            taken += 1
            if taken >= total_budget:
                break

        for field, indices in selected.items():
            indices.sort()
        return selected

    def _compute_metrics(
        self,
        encoded: Mapping[str, List[List[float]]],
        telemetry: CompressionTelemetry,
    ) -> Dict[str, float]:
        total = sum(len(v) for v in encoded.values())
        kept = sum(len(telemetry.selected_indices.get(field, [])) for field in encoded)
        utilisation = float(kept) / float(total) if total else 0.0
        metrics: Dict[str, float] = {
            "information_bound": utilisation,
            "rate_distortion": float(total - kept),
        }
        metrics.update(self._compute_information_proxies(encoded, telemetry))
        return metrics

    def _compute_mutual_information_lower_bound(
        self,
        encoded: Mapping[str, List[List[float]]],
        telemetry: CompressionTelemetry,
        mi_proxy: Optional["MIProxy"],
        targets: Optional[Mapping[str, Any]],
    ) -> Optional[float]:
        if mi_proxy is None:
            return None
        if torch is None:  # pragma: no cover - torch dependency enforced by proxy
            raise RuntimeError("torch is required when supplying an MI proxy")
        if not targets:
            return None

        samples: List["Tensor"] = []
        target_tensors: List["Tensor"] = []

        param = next(mi_proxy.parameters(), None)
        if param is None:  # pragma: no cover - module always defines parameters
            device = torch.device("cpu")
            dtype = torch.float32
        else:
            device = param.device
            dtype = param.dtype

        for field, raw_target in targets.items():
            indices = telemetry.selected_indices.get(field, [])
            if not indices:
                continue
            embeddings = encoded.get(field)
            if not embeddings:
                continue
            kept_vectors = [
                embeddings[i]
                for i in indices
                if 0 <= int(i) < len(embeddings)
            ]
            if not kept_vectors:
                continue
            pooled_vector = _mean_vector(kept_vectors)
            if not pooled_vector:
                continue
            sample_tensor = torch.as_tensor(pooled_vector, dtype=dtype, device=device)
            target_tensor = _coerce_tensor(raw_target, dtype, device)
            if target_tensor is None:
                continue
            sample_tensor = sample_tensor.view(-1)
            target_tensor = target_tensor.view(-1)
            if sample_tensor.size(0) != target_tensor.size(0):
                continue
            samples.append(sample_tensor)
            target_tensors.append(target_tensor)

        if not samples:
            return None

        z_tensor = torch.stack(samples, dim=0)
        target_tensor = torch.stack(target_tensors, dim=0)

        try:
            mi_value, _ = mi_proxy(z_tensor, target_tensor)
        except Exception:  # pragma: no cover - defensive safeguard
            return None

        if isinstance(mi_value, torch.Tensor):
            return float(mi_value.detach().cpu().item())
        try:
            return float(mi_value)
        except (TypeError, ValueError):  # pragma: no cover - unexpected return type
            return None

    def _compute_loss_terms(
        self,
        encoded: Mapping[str, List[List[float]]],
    ) -> Dict[str, Tensor]:
        if self.learnable_scorer is None or not self._score_tensors or torch is None:
            return {}
        base_tensor = next(iter(self._score_tensors.values()), None)
        if base_tensor is None:
            return {}
        device = base_tensor.device
        dtype = base_tensor.dtype
        kept_energy = torch.zeros(1, device=device, dtype=dtype)
        total_energy = torch.zeros(1, device=device, dtype=dtype)
        total_probs = torch.zeros(1, device=device, dtype=dtype)
        total_tokens = 0
        for field, logits in self._score_tensors.items():
            embeddings = encoded.get(field)
            if not embeddings:
                continue
            embedding_tensor = torch.as_tensor(embeddings, device=device, dtype=dtype)
            energy = (embedding_tensor * embedding_tensor).sum(dim=-1)
            probs = torch.sigmoid(logits)
            kept_energy = kept_energy + (probs * energy).sum()
            total_energy = total_energy + energy.sum()
            total_probs = total_probs + probs.sum()
            total_tokens += embedding_tensor.size(0)
        if total_tokens == 0:
            return {}
        eps = torch.finfo(dtype).eps
        denom = total_energy + eps
        ib_proxy = kept_energy / denom
        dropped_energy = total_energy - kept_energy
        token_normaliser = torch.tensor(float(total_tokens), device=device, dtype=dtype)
        rd_proxy = dropped_energy / (token_normaliser + eps)
        expected_keep_rate = total_probs / token_normaliser
        return {
            "ib_proxy": ib_proxy.squeeze(0),
            "rd_proxy": rd_proxy.squeeze(0),
            "expected_keep_rate": expected_keep_rate.squeeze(0),
        }

    def _compute_information_proxies(
        self,
        encoded: Mapping[str, List[List[float]]],
        telemetry: CompressionTelemetry,
    ) -> Dict[str, float]:
        kept_energy = 0.0
        dropped_energy = 0.0
        total_tokens = 0
        reconstruction_error = 0.0

        for field, embeddings in encoded.items():
            selected = set(telemetry.selected_indices.get(field, []))
            kept_vectors: List[Embedding] = []
            dropped_vectors: List[Embedding] = []
            for idx, vector in enumerate(embeddings):
                energy = _vector_energy(vector)
                if idx in selected:
                    kept_energy += energy
                    kept_vectors.append(vector)
                else:
                    dropped_energy += energy
                    dropped_vectors.append(vector)
            total_tokens += len(embeddings)
            if kept_vectors and dropped_vectors:
                kept_mean = _mean_vector(kept_vectors)
                dropped_mean = _mean_vector(dropped_vectors)
                reconstruction_error += _mse(kept_mean, dropped_mean)

        total_energy = kept_energy + dropped_energy
        ib_proxy = kept_energy / total_energy if total_energy else 0.0
        rd_proxy = dropped_energy / float(total_tokens or 1)
        average_reconstruction = (
            reconstruction_error / float(total_tokens or 1)
        )
        return {
            "ib_proxy": ib_proxy,
            "rd_proxy": rd_proxy,
            "embedding_reconstruction_error": average_reconstruction,
        }

    def _normalise_field_metadata(
        self,
        fields: Iterable[str],
        field_specs: Optional[Mapping[str, "FieldSpec | Mapping[str, Any]"]],
        registry: Optional["Registry"],
    ) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        combined: MutableMapping[str, Any] = {}
        if field_specs:
            combined.update(field_specs)
        if registry is not None:
            combined.update(getattr(registry, "fields", {}))
        for field in fields:
            metadata[field] = _coerce_field_metadata(combined.get(field))
        return metadata

    def _resolve_objective(self, objective: str) -> ScoringFn:
        key = objective.lower()
        if key in {"l2", "l2-norm", "norm", "magnitude"}:
            return cast(ScoringFn, NormScoringStrategy())
        if key in {"query", "query-dot", "query_attention", "attention"}:
            return QueryDotProductScoringStrategy()
        raise ValueError(f"unknown objective '{objective}'")


def _coerce_field_metadata(raw: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any]
    if raw is None:
        return {"keys": [], "salience": False, "metadata": {}, "budget_weight": None}

    if hasattr(raw, "keys") and hasattr(raw, "salience"):
        metadata_dict = dict(getattr(raw, "metadata", {}) or {})
        budget_weight = (
            metadata_dict.pop("budget_weight", None)
            if isinstance(metadata_dict, dict)
            else None
        )
        if budget_weight is not None:
            try:
                budget_weight = float(budget_weight)
            except (TypeError, ValueError):
                budget_weight = None
        return {
            "keys": list(getattr(raw, "keys", [])),
            "salience": bool(getattr(raw, "salience", False)),
            "metadata": metadata_dict,
            "budget_weight": budget_weight,
        }

    if isinstance(raw, Mapping):
        meta: Dict[str, Any] = {}
        nested = raw.get("metadata")
        if isinstance(nested, Mapping):
            meta.update(nested)
        for key, value in raw.items():
            if key not in {"keys", "salience", "metadata", "budget_weight"}:
                meta[key] = value
        budget_weight = raw.get("budget_weight")
        if budget_weight is None and "budget_weight" in meta:
            budget_weight = meta.pop("budget_weight")
        if isinstance(budget_weight, (int, float)):
            weight = float(budget_weight)
        else:
            try:
                weight = float(budget_weight)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                weight = None
        return {
            "keys": list(raw.get("keys", [])),
            "salience": bool(raw.get("salience", False)),
            "metadata": meta,
            "budget_weight": weight,
        }

    return {"keys": [], "salience": False, "metadata": {}, "budget_weight": None}


def _compute_residual_statistics(
    kept_embeddings: Sequence[Sequence[float]],
    dropped_embeddings: Sequence[Sequence[float]],
    scores: Sequence[float],
    kept_indices: Sequence[int],
    dropped_indices: Sequence[int],
) -> Dict[str, float]:
    kept_count = float(len(kept_embeddings))
    dropped_count = float(len(dropped_embeddings))
    if dropped_count == 0:
        return {
            "kept_count": kept_count,
            "dropped_count": 0.0,
            "mean_squared_error": 0.0,
            "kl_divergence": 0.0,
        }

    kept_mean = _mean_vector(kept_embeddings) if kept_embeddings else [0.0] * len(dropped_embeddings[0])
    dropped_mean = _mean_vector(dropped_embeddings)
    mse = _mse(kept_mean, dropped_mean)

    distribution = _softmax(scores)
    kept_probs = [_safe_probability(distribution, idx) for idx in kept_indices]
    dropped_probs = [_safe_probability(distribution, idx) for idx in dropped_indices]
    kl = _kl_divergence(kept_probs, dropped_probs)

    return {
        "kept_count": kept_count,
        "dropped_count": dropped_count,
        "mean_squared_error": float(mse),
        "kl_divergence": float(kl),
    }


def _quantize_embedding(vector: Sequence[float], levels: int = 256) -> Dict[str, Any]:
    if not vector:
        return {"values": [], "scale": 1.0}
    max_abs = max(abs(float(component)) for component in vector)
    if max_abs == 0.0:
        return {"values": [0 for _ in vector], "scale": 1.0}
    divisor = float((levels // 2) - 1 or 1)
    scale = max_abs / divisor if divisor else max_abs or 1.0
    quantized = [int(round(float(component) / scale)) for component in vector]
    return {"values": quantized, "scale": scale}


def _dequantize_embedding(entry: Mapping[str, Any]) -> List[float]:
    values_raw = entry.get("values", [])
    scale_raw = entry.get("scale", 1.0)
    try:
        scale = float(scale_raw)
    except (TypeError, ValueError):
        scale = 1.0
    if scale == 0.0:
        scale = 1.0
    if isinstance(values_raw, (list, tuple)):
        iterable = values_raw
    else:
        try:
            iterable = list(values_raw)  # type: ignore[arg-type]
        except TypeError:
            iterable = [values_raw]
    dequantized: List[float] = []
    for value in iterable:
        try:
            dequantized.append(float(int(value)) * scale)
        except (TypeError, ValueError):
            dequantized.append(0.0)
    return dequantized


def _softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(float(score) for score in scores)
    exps = [math.exp(float(score) - max_score) for score in scores]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / float(len(scores)) for _ in scores]
    return [value / total for value in exps]


def _safe_probability(distribution: Sequence[float], index: int) -> float:
    if 0 <= index < len(distribution):
        return float(distribution[index])
    return 0.0


def _kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-9) -> float:
    if not p or not q:
        return 0.0
    max_len = max(len(p), len(q))
    p_norm = list(p) + [0.0] * (max_len - len(p))
    q_norm = list(q) + [0.0] * (max_len - len(q))
    sum_p = sum(p_norm)
    sum_q = sum(q_norm)
    if sum_p == 0.0 or sum_q == 0.0:
        return 0.0
    kl = 0.0
    for pv, qv in zip(p_norm, q_norm):
        pv_norm = pv / sum_p
        qv_norm = qv / sum_q
        if pv_norm <= 0.0:
            continue
        kl += pv_norm * math.log(pv_norm / (qv_norm + eps))
    return kl


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(_vector_energy(vector)) if vector else 0.0


def _vector_energy(vector: Sequence[float]) -> float:
    return float(sum(float(component) * float(component) for component in vector))


def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dim = max(len(vector) for vector in vectors)
    accumulator = [0.0] * dim
    for vector in vectors:
        for idx in range(dim):
            value = float(vector[idx]) if idx < len(vector) else 0.0
            accumulator[idx] += value
    count = float(len(vectors))
    return [value / count for value in accumulator]


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    dim = max(len(a), len(b))
    if dim == 0:
        return 0.0
    error = 0.0
    for idx in range(dim):
        av = float(a[idx]) if idx < len(a) else 0.0
        bv = float(b[idx]) if idx < len(b) else 0.0
        diff = av - bv
        error += diff * diff
    return error / float(dim)


def _coerce_tensor(value: Any, dtype: "torch.dtype", device: "torch.device") -> Optional["Tensor"]:
    if torch is None:  # pragma: no cover - torch dependency handled by caller
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.to(device=device, dtype=dtype)
    if isinstance(value, Sequence):
        try:
            data = [float(component) for component in value]
        except (TypeError, ValueError):
            return None
        if not data:
            return None
        return torch.as_tensor(data, dtype=dtype, device=device)
    return None


__all__ = [
    "IBottleneck",
    "CompressionResult",
    "CompressionTelemetry",
    "NormScoringStrategy",
    "QueryDotProductScoringStrategy",
    "RegistryAwareBudgetAllocator",
]
