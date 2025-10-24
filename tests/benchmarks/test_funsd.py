from __future__ import annotations

from benchmarks.funsd import (
    build_funsd_encoders,
    build_funsd_registry,
    funsd_fields,
    funsd_numeric_answer_label,
    load_funsd_dataset,
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
