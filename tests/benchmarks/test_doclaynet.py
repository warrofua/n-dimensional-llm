from __future__ import annotations

from benchmarks.doclaynet import (
    build_doclaynet_encoders,
    build_doclaynet_registry,
    doclaynet_contains_table,
    doclaynet_fields,
    load_doclaynet_dataset,
)
from nd_llm.bottleneck import IBottleneck
from nd_llm.utils import build_mi_proxy_context


def test_doclaynet_sample_roundtrip() -> None:
    dataset = load_doclaynet_dataset(use_sample=True)
    assert dataset, "Expected bundled DocLayNet sample to be non-empty"

    registry = build_doclaynet_registry()
    build_doclaynet_encoders(registry)

    document = dataset[0]
    fields = doclaynet_fields(document)

    assert set(fields) == {"text", "layout", "region"}
    assert len(fields["text"]) == len(fields["layout"]) >= 1

    layout_by_token = {entry["token_id"]: entry for entry in fields["layout"]}
    regions_by_id = {entry["region_id"]: entry for entry in fields["region"]}

    for text_entry in fields["text"]:
        token_id = text_entry["token_id"]
        assert token_id in layout_by_token
        layout_entry = layout_by_token[token_id]
        assert layout_entry["region_id"] == text_entry["region_id"]
        region_id = text_entry["region_id"]
        if region_id in regions_by_id:
            assert token_id in regions_by_id[region_id].get("token_ids", [])

    assert all(0.0 <= coord <= 1.0 for item in fields["layout"] for coord in item.get("xyxy", []))
    assert isinstance(doclaynet_contains_table(document), bool)

    bottleneck = IBottleneck(target_budget=4)
    mi_proxy, mi_context = build_mi_proxy_context(
        fields,
        registry.encoders,
        preferred_fields=("layout", "text"),
    )
    result = bottleneck.compress(
        fields,
        encoders=registry.encoders,
        context=mi_context,
        mi_proxy=mi_proxy,
    )
    assert "region" in result.compressed_fields
    assert result.compressed_fields["region"], "Expected at least one region to be retained"


def test_doclaynet_dataset_limit() -> None:
    dataset = load_doclaynet_dataset(use_sample=True, limit=2)
    assert len(dataset) == 2
