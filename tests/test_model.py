from __future__ import annotations

import torch

from nd_llm.bottleneck import CompressionTelemetry
from nd_llm.encoders import LayoutEncoder, TextEncoder
from nd_llm.metrics import MIProxy
from nd_llm.model import NDEncoderDecoder
from nd_llm.utils import pack_fields


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
