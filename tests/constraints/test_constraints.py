from __future__ import annotations

from nd_llm.constraints import (
    FieldActivationConstraint,
    SuperpositionSimilarityConstraint,
)
from nd_llm.orchestration.orchestrator import CompressionRecord, UsageEvent
from nd_llm.stm import STM
from nd_llm.utils import STMConfig


def _make_record(selected_text: int) -> CompressionRecord:
    telemetry = {
        "selected_indices": {"text": list(range(selected_text))},
        "token_counts": {"text": selected_text},
    }
    return CompressionRecord(
        compressed_fields={"text": ["t"] * selected_text},
        telemetry=telemetry,
        metrics={},
    )


def test_field_activation_constraint_flags_underflow(tmp_path) -> None:
    stm = STM(STMConfig(storage_dir=tmp_path))
    constraint = FieldActivationConstraint(field="text", min_tokens=2, max_tokens=3)
    record = _make_record(1)
    event = UsageEvent(tensor=[[]], metadata={}, compression=record)

    result = constraint.evaluate(stm=stm, event=event, compression=record)
    assert result.name == "field_activation"
    assert result.satisfied is False
    assert result.details["count"] == 1


def test_superposition_similarity_constraint_reads_memory(tmp_path) -> None:
    stm = STM(STMConfig(storage_dir=tmp_path))
    stm.write_superposition("usage", [1.0, 0.0], metadata={"task": "demo"})
    constraint = SuperpositionSimilarityConstraint(channel="usage", min_similarity=0.5)
    record = _make_record(1)
    event = UsageEvent(tensor=[1.0, 0.0], metadata={}, compression=record)

    result = constraint.evaluate(stm=stm, event=event, compression=record)
    assert result.satisfied is True
    assert result.details["channel"] == "usage"
