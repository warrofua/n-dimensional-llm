from __future__ import annotations

import torch

from nd_llm.bottleneck import CompressionTelemetry
from nd_llm.encoders import LayoutEncoder, TextEncoder
from nd_llm.metrics import MIProxy
from nd_llm.model import NDEncoderDecoder


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
