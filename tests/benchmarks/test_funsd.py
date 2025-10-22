from __future__ import annotations

from benchmarks.funsd import (
    build_funsd_encoders,
    build_funsd_registry,
    funsd_fields,
    funsd_numeric_answer_label,
    load_funsd_dataset,
)
from nd_llm.bottleneck import IBottleneck


def test_funsd_sample_roundtrip() -> None:
    dataset = load_funsd_dataset(use_sample=True)
    assert dataset, "Expected bundled FUNSD sample to be non-empty"

    registry = build_funsd_registry()
    build_funsd_encoders(registry)

    document = dataset[0]
    fields = funsd_fields(document)

    assert set(fields) == {"text", "layout", "entity"}
    assert len(fields["text"]) == len(fields["layout"])
    assert all(0.0 <= coord <= 1.0 for item in fields["layout"] for coord in item.get("xyxy", []))

    result = IBottleneck(target_budget=6).compress(fields, encoders=registry.encoders)
    assert "text" in result.compressed_fields
    assert isinstance(funsd_numeric_answer_label(document), bool)
