"""Tests for STM query and alignment APIs."""

from __future__ import annotations

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
    }
    metrics = {"information_bound": kept_tokens / total_tokens if total_tokens else 0.0}
    return CompressionRecord(
        compressed_fields=compressed_fields,
        telemetry=telemetry,
        metrics=metrics,
        bottleneck="ib-topk",
    )


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
