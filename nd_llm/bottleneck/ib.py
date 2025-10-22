"""Information bottleneck stub with top-k gating."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from nd_llm.encoders import Encoder


@dataclass
class CompressionTelemetry:
    """Telemetry collected during compression for later analytics."""

    selected_indices: Dict[str, List[int]]
    selected_scores: Dict[str, List[float]]
    token_counts: Dict[str, int]
    budget: int


@dataclass
class CompressionResult:
    """Container returned by :meth:`IBottleneck.compress`."""

    compressed_fields: Dict[str, List[Any]]
    telemetry: CompressionTelemetry
    metrics: Dict[str, float]


class IBottleneck:
    """Simple top-k information bottleneck with placeholder IB/RD metrics."""

    def __init__(self, target_budget: int) -> None:
        if target_budget <= 0:
            raise ValueError("target_budget must be positive")
        self.target_budget = int(target_budget)

    def compress(
        self,
        fields: Mapping[str, Sequence[Any]],
        encoders: Mapping[str, Encoder],
    ) -> CompressionResult:
        if not fields:
            raise ValueError("fields must not be empty")

        encoded = self._encode_fields(fields, encoders)
        gating_scores = self._compute_scores(encoded)

        selected = self._select_indices(gating_scores)
        compressed_fields: Dict[str, List[Any]] = {}
        token_counts: Dict[str, int] = {}
        selected_scores: Dict[str, List[float]] = {}
        for field, indices in selected.items():
            source = list(fields[field])
            compressed_fields[field] = [source[i] for i in indices]
            token_counts[field] = len(source)
            selected_scores[field] = [gating_scores[field][i] for i in indices]

        telemetry = CompressionTelemetry(
            selected_indices={k: list(v) for k, v in selected.items()},
            selected_scores=selected_scores,
            token_counts=token_counts,
            budget=self.target_budget,
        )
        metrics = self._compute_metrics(token_counts, telemetry)
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

    def _compute_scores(self, encoded: Mapping[str, List[List[float]]]) -> Dict[str, List[float]]:
        scores: Dict[str, List[float]] = {}
        for field, embeddings in encoded.items():
            field_scores = [
                math.sqrt(sum(component * component for component in vector)) if vector else 0.0
                for vector in embeddings
            ]
            scores[field] = field_scores
        return scores

    def _select_indices(self, scores: Mapping[str, List[float]]) -> Dict[str, List[int]]:
        total_tokens = sum(len(v) for v in scores.values())
        if total_tokens == 0:
            return {field: [] for field in scores}
        budget = min(self.target_budget, total_tokens)
        flat: List[tuple[float, str, int]] = []
        for field, values in scores.items():
            for idx, score in enumerate(values):
                flat.append((score, field, idx))
        flat.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected_pairs = flat[:budget]
        selected: Dict[str, List[int]] = {field: [] for field in scores}
        for _, field, idx in selected_pairs:
            selected[field].append(idx)
        for field, indices in selected.items():
            indices.sort()
        return selected

    def _compute_metrics(
        self,
        token_counts: Mapping[str, int],
        telemetry: CompressionTelemetry,
    ) -> Dict[str, float]:
        total = sum(token_counts.values())
        kept = sum(len(v) for v in telemetry.selected_indices.values())
        utilisation = float(kept) / float(total) if total else 0.0
        return {
            "information_bound": utilisation,
            "rate_distortion": float(total - kept),
        }
