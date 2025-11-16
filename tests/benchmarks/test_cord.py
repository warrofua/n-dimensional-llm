from __future__ import annotations

import pytest

from benchmarks.cord import (
    build_cord_encoders,
    build_cord_registry,
    cord_amount_from_text,
    cord_fields,
    cord_high_total_label,
    cord_total_amount,
    load_cord_dataset,
)
from nd_llm.bottleneck import IBottleneck
from nd_llm.utils import build_mi_proxy_context
import json
from pathlib import Path


def test_cord_sample_roundtrip() -> None:
    dataset = load_cord_dataset(use_sample=True, limit=1)
    assert dataset, "Expected bundled CORD sample to be non-empty"

    registry = build_cord_registry()
    build_cord_encoders(registry)

    document = dataset[0]
    fields = cord_fields(document)

    assert set(fields) == {"text", "layout", "line"}
    assert len(fields["text"]) == len(fields["layout"])
    assert all(item["xyxy"] for item in fields["layout"])
    assert all(
        0.0 <= coord <= 1.0 for entry in fields["layout"] for coord in entry["xyxy"]
    )
    assert all(entry.get("coords") for entry in fields["text"])
    assert all(
        0.0 <= coord <= 1.0 for entry in fields["text"] for coord in entry["coords"]
    )
    assert all(entry.get("coords") for entry in fields["line"])

    mi_proxy, mi_context = build_mi_proxy_context(
        fields,
        registry.encoders,
        preferred_fields=("text", "line"),
    )
    bottleneck = IBottleneck(target_budget=4)
    result = bottleneck.compress(
        fields,
        encoders=registry.encoders,
        context=mi_context,
        mi_proxy=mi_proxy,
    )
    assert "text" in result.compressed_fields
    assert cord_total_amount(document) > 0
    assert isinstance(cord_high_total_label(document, threshold=5.0), bool)


def test_cord_amount_from_text_variants() -> None:
    assert cord_amount_from_text("1,234.56") == pytest.approx(1234.56)
    assert cord_amount_from_text("Total 59,000") == pytest.approx(59000.0)
    assert cord_amount_from_text("Grand Total 0") == 0.0


def test_load_cord_dataset_from_local_root(tmp_path) -> None:
    root = tmp_path / "CORD"
    split_dir = root / "train" / "json"
    split_dir.mkdir(parents=True)
    sample_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "data"
        / "cord_sample.jsonl"
    )
    sample_line = json.loads(sample_path.read_text().splitlines()[0])
    (split_dir / "sample.json").write_text(json.dumps(sample_line))

    dataset = load_cord_dataset(
        split="train", use_sample=False, data_root=root, limit=1
    )
    assert dataset and dataset[0]["doc_id"] == "cord-sample-0"
