"""Tests for STM query and alignment APIs."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nd_llm.orchestration import CompressionRecord
from nd_llm.stm import STM
from nd_llm.utils import STMConfig


def _make_record(total_tokens: int, kept_tokens: int, budget: int) -> CompressionRecord:
    compressed_fields = {"text": [f"tok_{i}" for i in range(kept_tokens)]}
    telemetry = {
        "selected_indices": {"text": list(range(kept_tokens))},
        "selected_scores": {"text": [1.0] * kept_tokens},
        "token_counts": {"text": total_tokens},
        "budget": budget,
        "dropped_indices": {"text": list(range(kept_tokens, total_tokens))},
    }
    metrics = {"information_bound": kept_tokens / total_tokens if total_tokens else 0.0}
    return CompressionRecord(
        compressed_fields=compressed_fields,
        telemetry=telemetry,
        metrics=metrics,
        bottleneck="ib-topk",
    )


def _make_canonical_record() -> CompressionRecord:
    compressed_fields = {
        "layout": [
            {
                "token": "alpha",
                "canonical_cell_id": "row-0-col-0",
                "embedding": [1.0, 0.0],
                "coords": [0.0, 0.0],
            },
            {
                "token": "beta",
                "canonical_cell_id": "row-0-col-1",
                "embedding": [0.5, 0.5],
                "coords": [0.0, 1.0],
            },
        ]
    }
    telemetry = {
        "selected_indices": {"layout": [0, 1]},
        "selected_scores": {"layout": [0.8, 0.75]},
        "token_counts": {"layout": 3},
        "budget": 2,
        "dropped_indices": {"layout": [2]},
        "cell_centers": [[[0.0, 0.0], [0.0, 1.0], [1.0, 0.5]]],
        "cell_agg": "mean",
    }
    metrics = {"mi_lb": 0.42}
    return CompressionRecord(
        compressed_fields=compressed_fields,
        telemetry=telemetry,
        metrics=metrics,
        bottleneck="ib-topk",
    )


def _layout_signature(idx_cells: dict[str, object]) -> str:
    canonical = idx_cells.get("canonical") if isinstance(idx_cells, dict) else None
    if not isinstance(canonical, dict):
        return ""
    parts: list[str] = []
    for field in sorted(canonical):
        values = canonical[field]
        if isinstance(values, (list, tuple, set, frozenset)):
            serialised = [str(item) for item in values if item is not None]
        elif values is None:
            serialised = []
        else:
            serialised = [str(values)]
        if serialised:
            parts.append(f"{field}:{','.join(serialised)}")
    if not parts:
        return ""
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def test_query_filters_by_alignment_and_ratio(tmp_path) -> None:
    stm = STM(STMConfig(storage_dir=tmp_path))

    record_low = _make_record(total_tokens=10, kept_tokens=4, budget=5)
    record_high = _make_record(total_tokens=10, kept_tokens=9, budget=9)

    stm.append(
        "entry-1",
        tensor=[0.1, 0.2],
        metadata={"alignment_key": "session-1", "compression": record_low.as_metadata()},
    )
    stm.append(
        "entry-2",
        tensor=[0.3, 0.4],
        metadata={"alignment_key": "session-2", "compression": record_high.as_metadata()},
    )

    session_two = stm.list_by_alignment("session-2")
    assert session_two == ["entry-2"]

    ratio_query = stm.query(metadata_filter={"compression.summary.compression_ratio": record_high.summary()["compression_ratio"]})
    keys = [key for key, _ in ratio_query]
    assert keys == ["entry-2"]

    empty = stm.query(metadata_filter={"alignment_key": "missing"})
    assert empty == []

    with pytest.raises(KeyError):
        stm.get_index_entry("unknown")

    index_entry = stm.get_index_entry("entry-1")
    assert index_entry["compression_summary"]["tokens_retained"] == pytest.approx(4.0)


def test_metadata_includes_canonical_cells(tmp_path) -> None:
    record = _make_canonical_record()
    metadata = record.as_metadata()

    assert metadata["pipeline"]["bottleneck"] == "ib-topk"
    assert metadata["K"] == 2
    assert metadata["mi_lb"] == pytest.approx(0.42)
    assert metadata["idx_cells"]["kept"]["layout"] == [0, 1]
    assert metadata["idx_cells"]["canonical"]["layout"] == [
        "row-0-col-0",
        "row-0-col-1",
    ]
    assert "canonical_cells" in metadata["artifacts"]

    stm = STM(STMConfig(storage_dir=tmp_path))
    layout_signature = _layout_signature(metadata["idx_cells"])
    stm.append(
        "doc-1",
        tensor=[0.0, 1.0],
        metadata={
            "task": "doc-vqa",
            "layout_signature": layout_signature,
            "compression": metadata,
        },
    )

    assert stm.list_by_task("doc-vqa") == ["doc-1"]
    if layout_signature:
        assert stm.list_by_layout(layout_signature) == ["doc-1"]
