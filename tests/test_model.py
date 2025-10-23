from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn

from nd_llm.bottleneck import CompressionResult, CompressionTelemetry
from nd_llm.encoders import LayoutEncoder, TextEncoder
from nd_llm.metrics import MIProxy
from nd_llm.model import NDEncoderDecoder
from nd_llm.utils import pack_fields


class RecordingMIProxy(nn.Module):
    """MI proxy stub that records the raw input tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(
        self, z: torch.Tensor, y_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        if not isinstance(y_repr, torch.Tensor):
            y_repr = torch.as_tensor(y_repr)
        self.last_input = z.detach().to(device="cpu")
        batch_size = z.size(0) if z.dim() >= 1 else 0
        logits = torch.zeros((batch_size, batch_size), device=z.device, dtype=z.dtype)
        mi_bound = torch.zeros((), device=z.device, dtype=z.dtype, requires_grad=True)
        return mi_bound, logits


class FakeBottleneck:
    """Deterministic bottleneck stub that returns predefined selections."""

    def __init__(self, selections: Mapping[str, Sequence[int]]) -> None:
        self._selections = {str(field): list(indices) for field, indices in selections.items()}
        total_selected = sum(len(indices) for indices in self._selections.values())
        self.target_budget = max(1, total_selected)

    def compress(
        self,
        fields: Mapping[str, Sequence[Any]],
        encoders: Mapping[str, Any],
        **_: Any,
    ) -> CompressionResult:
        selected_indices: Dict[str, list[int]] = {}
        selected_scores: Dict[str, list[float]] = {}
        compressed_fields: Dict[str, list[Any]] = {}
        token_counts: Dict[str, int] = {}
        dropped_indices: Dict[str, list[int]] = {}

        for field, items in fields.items():
            key = str(field)
            entries = list(items)
            indices = list(self._selections.get(key, []))
            selected_indices[key] = indices
            token_counts[key] = len(entries)
            selected_scores[key] = [float(i + 1) for i in range(len(indices))]
            compressed_fields[key] = [entries[idx] for idx in indices if 0 <= idx < len(entries)]
            selected_set = set(indices)
            dropped_indices[key] = [idx for idx in range(len(entries)) if idx not in selected_set]

        for field, indices in self._selections.items():
            if field not in selected_indices:
                selected_indices[field] = list(indices)
                selected_scores[field] = [float(i + 1) for i in range(len(indices))]
                compressed_fields[field] = []
                token_counts[field] = 0
                dropped_indices[field] = []

        telemetry = CompressionTelemetry(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            token_counts=token_counts,
            budget=self.target_budget,
            field_budgets={field: len(indices) for field, indices in selected_indices.items()},
            allocation_weights={field: 1.0 for field in selected_indices},
            dropped_indices=dropped_indices,
            residual_statistics={field: {} for field in selected_indices},
            quantized_embeddings={field: [] for field in selected_indices},
        )

        return CompressionResult(
            compressed_fields=compressed_fields,
            telemetry=telemetry,
            metrics={},
            loss_terms={},
        )


def test_nd_encoder_decoder_emits_compression_telemetry() -> None:
    torch.manual_seed(0)
    model = NDEncoderDecoder(hidden_dim=32, num_classes=2)
    assert isinstance(model.mi, MIProxy)
    text_encoder = TextEncoder(embedding_dim=8)
    layout_encoder = LayoutEncoder(embedding_dim=6)

    model.register_field(
        "text",
        encoder=text_encoder,
        keys=["doc_id", "span_id"],
        salience=True,
        value_key="text",
    )
    model.register_field(
        "layout",
        encoder=layout_encoder,
        keys=["doc_id", "span_id"],
    )

    fields = {
        "text": [
            {"doc_id": 0, "span_id": 0, "text": "alpha"},
            {"doc_id": 1, "span_id": 0, "text": "beta"},
            {"doc_id": 0, "span_id": 1, "text": "gamma"},
        ],
        "layout": [
            {"doc_id": 0, "span_id": 0, "xyxy": (0.0, 0.0, 1.0, 1.0)},
            {"doc_id": 1, "span_id": 0, "xyxy": (1.0, 1.0, 2.0, 2.0)},
            {"doc_id": 0, "span_id": 1, "xyxy": (0.5, 0.5, 1.5, 1.5)},
        ],
    }
    doc_ids = [0, 1]
    batch = {
        "fields": fields,
        "doc_ids": doc_ids,
        "targets": torch.tensor([1, 0], dtype=torch.long),
    }

    logits, logs = model(batch, token_budget=3)

    assert logits.shape[0] == len(doc_ids)
    telemetry = logs.get("compression_telemetry")
    assert isinstance(telemetry, CompressionTelemetry)
    assert any(len(indices) for indices in telemetry.selected_indices.values())
    metrics = logs.get("compression_metrics")
    assert isinstance(metrics, dict)
    assert metrics.get("information_bound", 0.0) > 0.0
    token_mask = logs.get("token_mask")
    assert isinstance(token_mask, torch.Tensor) and token_mask.dtype == torch.bool
    tokens_selected = logs.get("tokens_selected")
    assert isinstance(tokens_selected, torch.Tensor)
    assert token_mask.sum().item() == int(tokens_selected.sum().item())
    selected_metadata = logs.get("selected_metadata")
    assert isinstance(selected_metadata, list)
    assert len(selected_metadata) == len(doc_ids)


def test_nd_encoder_decoder_handles_missing_targets() -> None:
    torch.manual_seed(0)
    model = NDEncoderDecoder(hidden_dim=32, num_classes=3)
    text_encoder = TextEncoder(embedding_dim=8)
    model.register_field(
        "text",
        encoder=text_encoder,
        keys=["doc_id", "span_id"],
        value_key="text",
    )

    fields = {
        "text": [
            {"doc_id": 0, "span_id": 0, "text": "alpha"},
            {"doc_id": 1, "span_id": 0, "text": "beta"},
        ]
    }
    doc_ids = [0, 1]
    batch = {
        "fields": fields,
        "doc_ids": doc_ids,
    }

    logits, logs = model(batch, token_budget=2)

    assert logits.shape == (len(doc_ids), model.num_classes)
    assert "targets" not in logs
    assert "target_repr" not in logs
    mi_lb_tensor = logs.get("mi_lb_tensor")
    assert isinstance(mi_lb_tensor, torch.Tensor)
    assert mi_lb_tensor.ndim == 0
    assert mi_lb_tensor.item() == 0.0
    assert logs.get("mi_lb") == 0.0


def test_nd_encoder_decoder_zero_token_budget_has_zero_mi_lb() -> None:
    torch.manual_seed(0)
    model = NDEncoderDecoder(hidden_dim=32, num_classes=2)
    text_encoder = TextEncoder(embedding_dim=8)
    layout_encoder = LayoutEncoder(embedding_dim=6)

    model.register_field(
        "text",
        encoder=text_encoder,
        keys=["doc_id", "span_id"],
        salience=True,
        value_key="text",
    )
    model.register_field(
        "layout",
        encoder=layout_encoder,
        keys=["doc_id", "span_id"],
    )

    fields = {
        "text": [
            {"doc_id": 0, "span_id": 0, "text": "alpha"},
            {"doc_id": 1, "span_id": 0, "text": "beta"},
            {"doc_id": 0, "span_id": 1, "text": "gamma"},
        ],
        "layout": [
            {"doc_id": 0, "span_id": 0, "xyxy": (0.0, 0.0, 1.0, 1.0)},
            {"doc_id": 1, "span_id": 0, "xyxy": (1.0, 1.0, 2.0, 2.0)},
            {"doc_id": 0, "span_id": 1, "xyxy": (0.5, 0.5, 1.5, 1.5)},
        ],
    }
    doc_ids = [0, 1]
    batch = {
        "fields": fields,
        "doc_ids": doc_ids,
        "targets": torch.tensor([1, 0], dtype=torch.long),
    }

    logits, logs = model(batch, token_budget=0)

    assert logits.shape[0] == len(doc_ids)
    mi_lb = logs.get("mi_lb")
    assert isinstance(mi_lb, float)
    assert mi_lb == 0.0
    mi_lb_tensor = logs.get("mi_lb_tensor")
    assert isinstance(mi_lb_tensor, torch.Tensor)
    assert torch.isfinite(mi_lb_tensor).item()
    assert mi_lb_tensor.requires_grad
    assert float(mi_lb_tensor.item()) == 0.0


def test_nd_encoder_decoder_mi_proxy_receives_masked_means() -> None:
    torch.manual_seed(0)
    selections = {"text": [0, 1, 2]}
    bottleneck = FakeBottleneck(selections)
    recording_proxy = RecordingMIProxy()
    model = NDEncoderDecoder(
        hidden_dim=32,
        num_classes=3,
        bottleneck=bottleneck,
        mi_proxy=recording_proxy,
    )

    text_encoder = TextEncoder(embedding_dim=8)
    model.register_field(
        "text",
        encoder=text_encoder,
        keys=["doc_id", "span_id"],
        value_key="text",
    )

    fields = {
        "text": [
            {"doc_id": 0, "span_id": 0, "text": "alpha"},
            {"doc_id": 0, "span_id": 1, "text": "beta"},
            {"doc_id": 1, "span_id": 0, "text": "gamma"},
            {"doc_id": 2, "span_id": 0, "text": "delta"},
        ]
    }
    doc_ids = [0, 1, 2]
    targets = torch.tensor([1, 0, 2], dtype=torch.long)
    batch = {"fields": fields, "doc_ids": doc_ids, "targets": targets}

    _, logs = model(batch, token_budget=3)

    recorded = recording_proxy.last_input
    assert recorded is not None
    assert recorded.dim() == 2
    assert recorded.shape[0] == len(doc_ids)

    aggregated = model.aggregator.aggregate(fields, doc_ids=doc_ids)
    telemetry = logs.get("compression_telemetry")
    assert isinstance(telemetry, CompressionTelemetry)

    selected_tokens, selected_mask, *_ = model._gather_selected_tokens(
        aggregated.tokens,
        aggregated.mask,
        aggregated.metadata,
        telemetry,
    )

    mask = selected_mask.unsqueeze(-1).float()
    masked_sums = (selected_tokens * mask).sum(dim=1)
    mask_counts = mask.sum(dim=1)
    expected = torch.where(
        mask_counts > 0,
        masked_sums / mask_counts.clamp_min(1.0),
        torch.zeros_like(masked_sums),
    )

    recorded_cpu = recorded.to(device="cpu")
    mask_counts_cpu = mask_counts.squeeze(-1).to(device="cpu")
    expected_cpu = expected.to(device="cpu")

    torch.testing.assert_close(recorded_cpu, expected_cpu)

    has_tokens = mask_counts_cpu > 0
    assert torch.all(recorded_cpu[has_tokens].abs().sum(dim=1) > 0)
    no_tokens = ~has_tokens
    if torch.any(no_tokens):
        zeros = torch.zeros_like(recorded_cpu[no_tokens])
        torch.testing.assert_close(recorded_cpu[no_tokens], zeros)


def test_canonical_cell_aggregator_preserves_all_tokens() -> None:
    torch.manual_seed(0)
    model = NDEncoderDecoder(hidden_dim=16, num_classes=2)
    text_encoder = TextEncoder(embedding_dim=8)
    layout_encoder = LayoutEncoder(embedding_dim=6)

    model.register_field(
        "text",
        encoder=text_encoder,
        keys=["doc_id", "span_id"],
        value_key="text",
    )
    model.register_field(
        "layout",
        encoder=layout_encoder,
        keys=["doc_id", "span_id"],
    )

    fields = {
        "text": [
            {"doc_id": 0, "span_id": 0, "text": "alpha"},
            {"doc_id": 0, "span_id": 1, "text": "beta"},
            {"doc_id": 1, "span_id": 0, "text": "gamma"},
            {"doc_id": 1, "span_id": 1, "text": "delta"},
        ],
        "layout": [
            {"doc_id": 0, "span_id": 0, "xyxy": (0.0, 0.0, 1.0, 1.0)},
            {"doc_id": 0, "span_id": 1, "xyxy": (0.1, 0.1, 1.1, 1.1)},
            {"doc_id": 1, "span_id": 0, "xyxy": (1.0, 1.0, 2.0, 2.0)},
            {"doc_id": 1, "span_id": 1, "xyxy": (1.1, 1.1, 2.1, 2.1)},
        ],
    }
    doc_ids = [0, 1]

    aggregated = model.aggregator.aggregate(fields, doc_ids=doc_ids)

    assert aggregated.tokens.shape[0] == len(doc_ids)
    assert aggregated.mask.shape == aggregated.tokens.shape[:2]
    assert aggregated.token_counts == [4, 4]
    for doc_value, metadata in zip(aggregated.doc_values, aggregated.metadata):
        assert doc_value in (0, 1)
        assert len(metadata) == 4


def test_nd_encoder_decoder_zero_budget_yields_zero_mi() -> None:
    torch.manual_seed(0)
    model = NDEncoderDecoder(hidden_dim=32, num_classes=3)
    text_encoder = TextEncoder(embedding_dim=8)
    model.register_field(
        "text",
        encoder=text_encoder,
        keys=["doc_id", "span_id"],
        value_key="text",
    )

    fields = {
        "text": [
            {"doc_id": 0, "span_id": 0, "text": "alpha"},
            {"doc_id": 1, "span_id": 0, "text": "beta"},
        ]
    }
    doc_ids = [0, 1]
    batch = {
        "fields": fields,
        "doc_ids": doc_ids,
        "targets": torch.tensor([1, 2], dtype=torch.long),
    }

    logits, logs = model(batch, token_budget=0)

    assert logits.shape == (len(doc_ids), model.num_classes)
    mi_lb_tensor = logs.get("mi_lb_tensor")
    assert isinstance(mi_lb_tensor, torch.Tensor)
    assert mi_lb_tensor.ndim == 0
    assert mi_lb_tensor.item() == 0.0
    assert logs.get("mi_lb") == 0.0
    tokens_selected = logs.get("tokens_selected")
    assert isinstance(tokens_selected, torch.Tensor)
    assert tokens_selected.sum().item() == 0.0


def test_nd_encoder_decoder_accepts_packed_fields() -> None:
    torch.manual_seed(0)
    model = NDEncoderDecoder(hidden_dim=16, num_classes=2)
    text_encoder = TextEncoder(embedding_dim=8)
    layout_encoder = LayoutEncoder(embedding_dim=6)

    model.register_field(
        "text",
        encoder=text_encoder,
        keys=["doc_id", "span_id"],
        value_key="text",
    )
    model.register_field(
        "layout",
        encoder=layout_encoder,
        keys=["doc_id", "span_id"],
    )

    fields = {
        "text": [
            {"doc_id": 0, "span_id": 0, "text": "alpha"},
            {"doc_id": 0, "span_id": 1, "text": "beta"},
            {"doc_id": 1, "span_id": 0, "text": "gamma"},
            {"doc_id": 1, "span_id": 1, "text": "delta"},
        ],
        "layout": [
            {"doc_id": 0, "span_id": 0, "xyxy": (0.0, 0.0, 1.0, 1.0)},
            {"doc_id": 0, "span_id": 1, "xyxy": (0.1, 0.1, 1.1, 1.1)},
            {"doc_id": 1, "span_id": 0, "xyxy": (1.0, 1.0, 2.0, 2.0)},
            {"doc_id": 1, "span_id": 1, "xyxy": (1.1, 1.1, 2.1, 2.1)},
        ],
    }
    doc_ids = [0, 1]
    packed = pack_fields(**fields)

    targets = torch.tensor([1, 0], dtype=torch.long)
    raw_batch = {
        "fields": fields,
        "doc_ids": doc_ids,
        "targets": targets,
    }
    packed_batch = {
        "fields": packed,
        "doc_ids": doc_ids,
        "targets": targets,
    }

    _, raw_logs = model(raw_batch, token_budget=0)
    _, packed_logs = model(packed_batch, token_budget=0)

    assert raw_logs["doc_order"] == packed_logs["doc_order"]

    raw_metadata = raw_logs["cell_metadata"]
    packed_metadata = packed_logs["cell_metadata"]

    raw_counts = [len(items) for items in raw_metadata]
    packed_counts = [len(items) for items in packed_metadata]
    assert raw_counts == packed_counts

    raw_order = [[(entry["field"], entry["index"]) for entry in items] for items in raw_metadata]
    packed_order = [
        [(entry["field"], entry["index"]) for entry in items] for items in packed_metadata
    ]
    assert raw_order == packed_order
