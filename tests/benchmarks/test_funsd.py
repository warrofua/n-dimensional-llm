from __future__ import annotations

import pytest

from benchmarks.funsd import (
    build_funsd_encoders,
    build_funsd_registry,
    funsd_fields,
    funsd_numeric_answer_label,
    load_funsd_dataset,
    _normalise_box,
    _resolve_size,
)
from nd_llm.bottleneck import IBottleneck
from nd_llm.utils import build_mi_proxy_context


def test_funsd_sample_roundtrip() -> None:
    dataset = load_funsd_dataset(use_sample=True, limit=3)
    assert dataset, "Expected bundled FUNSD sample to be non-empty"

    registry = build_funsd_registry()
    build_funsd_encoders(registry)

    document = dataset[0]
    fields = funsd_fields(document)

    assert set(fields) == {"text", "layout", "entity"}
    assert len(fields["text"]) == len(fields["layout"])
    assert all(0.0 <= coord <= 1.0 for item in fields["layout"] for coord in item.get("xyxy", []))

    bottleneck = IBottleneck(target_budget=6)
    mi_proxy, mi_context = build_mi_proxy_context(
        fields,
        registry.encoders,
        preferred_fields=("text",),
    )
    result = bottleneck.compress(
        fields,
        encoders=registry.encoders,
        context=mi_context,
        mi_proxy=mi_proxy,
    )
    assert "text" in result.compressed_fields
    assert isinstance(funsd_numeric_answer_label(document), bool)


def test_normalise_box_xyxy_passthrough() -> None:
    result = _normalise_box([10, 20, 50, 60], width=100, height=100)
    assert result == [0.1, 0.2, 0.5, 0.6]


def test_normalise_box_xywh_with_mode() -> None:
    result = _normalise_box([10, 20, 30, 40], width=200, height=200, mode="xywh")
    assert result == [0.05, 0.1, 0.2, 0.3]


def test_normalise_box_xywh_inferred() -> None:
    result = _normalise_box([100, 120, 10, 20], width=200, height=200)
    assert result == [0.5, 0.6, 0.55, 0.7]


def test_resolve_size_reads_image(tmp_path) -> None:
    pytest.importorskip("PIL.Image")
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
            b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
        )
    )
    width, height = _resolve_size({"image": str(image_path)})
    assert (width, height) == (1.0, 1.0)
