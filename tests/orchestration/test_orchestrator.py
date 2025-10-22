"""Integration-style tests for the orchestration layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nd_llm.orchestration import (
    CompressionRecord,
    CompressionRatioBudgetStrategy,
    Orchestrator,
    UsageEvent,
)
from nd_llm.stm import STM
from nd_llm.utils import OrchestratorConfig, STMConfig


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _make_record(total_tokens: int, kept_tokens: int, budget: int, field: str = "text") -> CompressionRecord:
    assert kept_tokens <= total_tokens
    compressed_fields = {field: [f"{field}_{i}" for i in range(kept_tokens)]}
    telemetry = {
        "selected_indices": {field: list(range(kept_tokens))},
        "selected_scores": {field: [1.0] * kept_tokens},
        "token_counts": {field: total_tokens},
        "budget": budget,
    }
    metrics = {"information_bound": kept_tokens / total_tokens if total_tokens else 0.0}
    return CompressionRecord(
        compressed_fields=compressed_fields,
        telemetry=telemetry,
        metrics=metrics,
        bottleneck="ib-topk",
    )


def test_orchestrator_persists_usage_event(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)
    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(target_budget=1.0, policy_name="integration-test", budget_step=0.2),
    )

    tensor_values = [0.1, 0.5, 0.9]
    event = UsageEvent(
        key="event-1",
        tensor=tensor_values,
        metadata={"input_tokens": 10, "output_tokens": 3, "model": "demo"},
    )

    stored_key = orchestrator.log_usage_event(event)

    assert stored_key == "event-1"
    assert stored_key in stm

    retrieved_tensor, retrieved_metadata = stm.retrieve(stored_key)
    assert _as_list(retrieved_tensor) == pytest.approx(tensor_values)
    assert retrieved_metadata["input_tokens"] == 10
    assert retrieved_metadata["output_tokens"] == 3
    assert retrieved_metadata["policy_name"] == "integration-test"
    assert retrieved_metadata["target_budget"] == orchestrator.config.target_budget
    assert "timestamp" in retrieved_metadata

    sweep_results = orchestrator.budget_sweep([0.5, 1.0, 1.5])
    assert sweep_results[0.5]["within_budget"] is True
    assert sweep_results[1.5]["within_budget"] is False

    probe = orchestrator.run_retention_probe()
    assert probe["total_retained"] == 1
    assert probe["sampled_keys"] == []
    assert probe["log_size"] == 1
    assert probe["reconstruction"]["sample_size"] == 0
    assert probe["issues"] and probe["issues"][0]["issue"] == "missing_compression_metadata"


def test_orchestrator_generates_unique_keys(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)
    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(target_budget=0.5, policy_name="auto-key", budget_step=0.1),
    )

    tensor_values = [1.0, 2.0]

    key_one = orchestrator.log_usage_event(UsageEvent(tensor=tensor_values))
    key_two = orchestrator.log_usage_event(UsageEvent(tensor=tensor_values))

    assert key_one != key_two
    assert len(orchestrator.usage_log) == 2

    tensor_one, _ = stm.retrieve(key_one)
    tensor_two, _ = stm.retrieve(key_two)
    assert _as_list(tensor_one) == pytest.approx(tensor_values)
    assert _as_list(tensor_two) == pytest.approx(tensor_values)


def test_budget_tuning_updates_target_budget(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)
    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(
            target_budget=3.0,
            policy_name="adaptive",
            budget_step=0.25,
            retention_probe_sample_size=5,
        ),
    )

    high_util_record = _make_record(total_tokens=10, kept_tokens=9, budget=9)
    low_util_record = _make_record(total_tokens=10, kept_tokens=4, budget=5)

    orchestrator.log_usage_event(
        UsageEvent(
            key="session-1",
            tensor=[0.1, 0.2],
            metadata={"alignment_key": "session"},
            compression=high_util_record,
        )
    )
    orchestrator.log_usage_event(
        UsageEvent(
            key="session-2",
            tensor=[0.2, 0.4],
            metadata={"alignment_key": "session"},
            compression=high_util_record,
        )
    )

    strategy = CompressionRatioBudgetStrategy(lower_bound=0.6, upper_bound=0.8, step=0.5, min_budget=0.5)
    reduced_budget = orchestrator.tune_budget(strategy=strategy)
    assert reduced_budget < 3.0
    assert orchestrator.config.target_budget == pytest.approx(reduced_budget)
    assert orchestrator.budget_history[-1].reason == "over-utilised"

    orchestrator.log_usage_event(
        UsageEvent(
            key="session-3",
            tensor=[0.3, 0.6],
            metadata={"alignment_key": "session"},
            compression=low_util_record,
        )
    )

    increased_budget = orchestrator.tune_budget(strategy=strategy)
    assert increased_budget >= reduced_budget
    assert len(orchestrator.budget_history) >= 2
    assert orchestrator.budget_history[-1].metadata["average_ratio"] <= orchestrator.budget_history[-2].metadata["average_ratio"]


def test_retention_probe_reports_reconstruction(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)
    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(
            target_budget=5.0,
            policy_name="retention",
            budget_step=0.5,
            retention_probe_sample_size=5,
        ),
    )

    record_one = _make_record(total_tokens=8, kept_tokens=4, budget=4, field="text")
    record_two = _make_record(total_tokens=6, kept_tokens=6, budget=6, field="vision")

    key_one = orchestrator.log_usage_event(
        UsageEvent(
            key="probe-1",
            tensor=[1.0, 2.0, 3.0],
            metadata={"alignment_key": "user-1"},
            compression=record_one,
        )
    )
    key_two = orchestrator.log_usage_event(
        UsageEvent(
            key="probe-2",
            tensor=[3.0, 2.0, 1.0],
            metadata={"alignment_key": "user-2"},
            compression=record_two,
        )
    )

    probe = orchestrator.run_retention_probe()
    assert set(probe["sampled_keys"]) == {key_one, key_two}
    assert probe["issues"] == []
    reconstruction = probe["reconstruction"]
    assert reconstruction["sample_size"] == 2
    assert 0.0 < reconstruction["mean_quality"] <= 1.0
