from __future__ import annotations

from pathlib import Path

import pytest

from nd_llm import (
    CompressionRecord,
    IBottleneck,
    Orchestrator,
    Registry,
    STM,
    UsageEvent,
    pack_fields,
)
from nd_llm.encoders import LayoutEncoder, TextEncoder
from nd_llm.utils import build_mi_proxy_context


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def test_quickstart_workflow(tmp_path: Path) -> None:
    registry = Registry()
    registry.add_field("text", keys=["doc_id", "span_id"], salience=True)
    registry.add_field("bbox", keys=["doc_id", "span_id"])

    text_encoder = TextEncoder()
    layout_encoder = LayoutEncoder()
    registry.register_encoder("text", text_encoder)
    registry.register_encoder("bbox", layout_encoder)

    spans = [("hello", True), ("world", False)]
    boxes = [(0, 0, 1, 1), (1, 1, 2, 2)]

    fields = pack_fields(
        text=[
            {"doc_id": 1, "span_id": idx, "text": text, "salience": salience}
            for idx, (text, salience) in enumerate(spans)
        ],
        bbox=[
            {"doc_id": 1, "span_id": idx, "xyxy": box}
            for idx, box in enumerate(boxes)
        ],
    )

    assert fields.key_names == ["doc_id", "span_id"]
    assert fields.key_rows[0] == {"doc_id": 1, "span_id": 0}

    batches = fields.to_field_batches({"text": "text"})
    assert batches["text"] == ["hello", "world"]
    assert batches["bbox"][0] == boxes[0]

    bottleneck = IBottleneck(target_budget=2)
    mi_proxy, mi_context = build_mi_proxy_context(
        batches,
        registry.encoders,
        preferred_fields=("text",),
    )
    result = bottleneck.compress(
        batches,
        registry.encoders,
        context=mi_context,
        mi_proxy=mi_proxy,
    )
    assert set(result.compressed_fields) == {"text", "bbox"}

    stm = STM.from_path(tmp_path / "stm-store")
    orchestrator = Orchestrator.from_components(
        stm=stm,
        bottleneck=bottleneck,
        target_budget=1.0,
        policy_name="quickstart",
    )

    usage_key = orchestrator.log_usage_event(
        UsageEvent(
            tensor=result.telemetry.selected_scores.get("text", []),
            metadata=fields.key_rows[0],
            compression=CompressionRecord.from_result(result, bottleneck=bottleneck),
        )
    )

    assert usage_key in stm
    tensor, metadata = stm.retrieve(usage_key)
    assert _as_list(tensor) == pytest.approx(result.telemetry.selected_scores.get("text", []))
    assert metadata["policy_name"] == "quickstart"
    assert metadata["target_budget"] == orchestrator.config.target_budget
    assert metadata["compression"]["summary"]["tokens_retained"] <= metadata["compression"]["summary"]["tokens_total"]


def test_pack_fields_validates_alignment() -> None:
    with pytest.raises(ValueError):
        pack_fields(
            text=[{"doc_id": 1, "span_id": 0, "text": "alpha"}],
            bbox=[],
        )

    with pytest.raises(ValueError):
        pack_fields(
            text=[{"doc_id": 1, "span_id": 0, "text": "alpha"}],
            bbox=[{"doc_id": 1, "xyxy": (0, 0, 1, 1)}],
        )
