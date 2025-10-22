"""Information bottleneck with pluggable scoring and telemetry."""
from __future__ import annotations

import math
from dataclasses import dataclass
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
)

from nd_llm.encoders import Encoder

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from nd_llm.registry import FieldSpec, Registry

ScoreVector = List[float]
Embedding = Sequence[float]
EmbeddingBatch = Sequence[Embedding]
FieldMetadata = Mapping[str, Any]
ScoringFn = Callable[[str, EmbeddingBatch, FieldMetadata, Mapping[str, Any]], ScoreVector]
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


@dataclass
class CompressionResult:
    """Container returned by :meth:`IBottleneck.compress`."""

    compressed_fields: Dict[str, List[Any]]
    telemetry: CompressionTelemetry
    metrics: Dict[str, float]


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
        self.fallback = fallback or NormScoringStrategy()
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
        fallback = self.fallback or NormScoringStrategy()
        base_scores = fallback(field, embeddings, metadata, context)
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
        budget_allocator: Optional[BudgetAllocatorFn] = None,
    ) -> None:
        if target_budget <= 0:
            raise ValueError("target_budget must be positive")
        self.target_budget = int(target_budget)
        self.objective = (objective or "l2-norm").lower()
        self.scorer = scorer or self._resolve_objective(self.objective)
        self.budget_allocator = budget_allocator or RegistryAwareBudgetAllocator()

    def compress(
        self,
        fields: Mapping[str, Sequence[Any]],
        encoders: Mapping[str, Encoder],
        *,
        registry: Optional["Registry"] = None,
        field_specs: Optional[Mapping[str, "FieldSpec | Mapping[str, Any]"]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> CompressionResult:
        if not fields:
            raise ValueError("fields must not be empty")
        metadata = self._normalise_field_metadata(fields.keys(), field_specs, registry)

        encoded = self._encode_fields(fields, encoders)
        scoring_context = context or {}
        gating_scores = self._compute_scores(encoded, metadata, scoring_context)

        field_budgets, allocation_weights = self.budget_allocator(gating_scores, metadata, self.target_budget)
        selected = self._select_indices(gating_scores, field_budgets)

        compressed_fields: Dict[str, List[Any]] = {}
        token_counts: Dict[str, int] = {}
        selected_scores: Dict[str, List[float]] = {}
        dropped_indices: Dict[str, List[int]] = {}
        for field, indices in selected.items():
            source = list(fields[field])
            compressed_fields[field] = [source[i] for i in indices]
            token_counts[field] = len(source)
            field_scores = gating_scores[field]
            selected_scores[field] = [field_scores[i] for i in indices]
            selected_set = set(indices)
            dropped = sorted(i for i in range(len(field_scores)) if i not in selected_set)
            dropped_indices[field] = dropped

        telemetry = CompressionTelemetry(
            selected_indices={k: list(v) for k, v in selected.items()},
            selected_scores=selected_scores,
            token_counts=token_counts,
            budget=self.target_budget,
            field_budgets=field_budgets,
            allocation_weights=allocation_weights,
            dropped_indices=dropped_indices,
        )
        metrics = self._compute_metrics(encoded, telemetry)
        return CompressionResult(
            compressed_fields=compressed_fields,
            telemetry=telemetry,
            metrics=metrics,
        )

    def decompress(self, result: CompressionResult) -> Dict[str, List[Any]]:
        """For the stub the decompression simply returns the kept tokens."""

        return result.compressed_fields

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
            field_scores = self.scorer(field, embeddings, metadata.get(field, {}), context)
            if len(field_scores) != len(embeddings):
                raise ValueError(
                    f"scoring strategy returned {len(field_scores)} scores for {len(embeddings)} embeddings in field '{field}'"
                )
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
            strategy = NormScoringStrategy()
        elif key in {"query", "query-dot", "query_attention", "attention"}:
            strategy = QueryDotProductScoringStrategy()
        else:
            raise ValueError(f"unknown objective '{objective}'")
        return strategy


def _coerce_field_metadata(raw: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any]
    if raw is None:
        return {"keys": [], "salience": False, "metadata": {}, "budget_weight": None}

    if hasattr(raw, "keys") and hasattr(raw, "salience"):
        meta = dict(getattr(raw, "metadata", {}) or {})
        budget_weight = meta.pop("budget_weight", None) if isinstance(meta, dict) else None
        if budget_weight is not None:
            try:
                budget_weight = float(budget_weight)
            except (TypeError, ValueError):
                budget_weight = None
        return {
            "keys": list(getattr(raw, "keys", [])),
            "salience": bool(getattr(raw, "salience", False)),
            "metadata": meta,
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


__all__ = [
    "IBottleneck",
    "CompressionResult",
    "CompressionTelemetry",
    "NormScoringStrategy",
    "QueryDotProductScoringStrategy",
    "RegistryAwareBudgetAllocator",
]
