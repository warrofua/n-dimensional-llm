"""Integration-style tests for the orchestration layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nd_llm.orchestration import Orchestrator, UsageEvent
from nd_llm.stm import STM
from nd_llm.utils import OrchestratorConfig, STMConfig


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


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
    assert stored_key in probe["sampled_keys"]
    assert probe["log_size"] == 1


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
