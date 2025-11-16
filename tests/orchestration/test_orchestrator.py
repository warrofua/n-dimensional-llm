"""Integration-style tests for the orchestration layer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nd_llm.bottleneck import CompressionResult, CompressionTelemetry
from nd_llm.constraints import FieldActivationConstraint
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


def _make_record(
    total_tokens: int,
    kept_tokens: int,
    budget: int,
    field: str = "text",
    field_mi: Optional[float] = None,
) -> CompressionRecord:
    assert kept_tokens <= total_tokens
    compressed_fields = {field: [f"{field}_{i}" for i in range(kept_tokens)]}
    telemetry = {
        "selected_indices": {field: list(range(kept_tokens))},
        "selected_scores": {field: [1.0] * kept_tokens},
        "token_counts": {field: total_tokens},
        "budget": budget,
        "dropped_indices": {field: list(range(kept_tokens, total_tokens))},
    }
    if field_mi is not None:
        telemetry["field_mutual_information"] = {field: field_mi}
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
                "canonical_cell_id": "r0c0",
                "embedding": [1.0, 0.0],
                "coords": [0.0, 0.0],
            },
            {
                "token": "beta",
                "canonical_cell_id": "r0c1",
                "embedding": [0.5, 0.5],
                "coords": [0.0, 1.0],
            },
        ]
    }
    telemetry = {
        "selected_indices": {"layout": [0, 1]},
        "selected_scores": {"layout": [0.7, 0.6]},
        "token_counts": {"layout": 3},
        "budget": 2,
        "dropped_indices": {"layout": [2]},
        "cell_centers": [[[0.0, 0.0], [0.0, 1.0], [1.0, 0.5]]],
    }
    metrics = {"mi_lb": 0.25}
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
        config=OrchestratorConfig(
            target_budget=1.0, policy_name="integration-test", budget_step=0.2
        ),
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
    assert (
        probe["issues"]
        and probe["issues"][0]["issue"] == "missing_compression_metadata"
    )


def test_orchestrator_generates_unique_keys(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)
    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(
            target_budget=0.5, policy_name="auto-key", budget_step=0.1
        ),
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


def test_orchestrator_enriches_layout_metadata(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)
    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(
            target_budget=2.0, policy_name="layout", budget_step=0.5
        ),
    )

    record = _make_canonical_record()
    key = orchestrator.log_usage_event(
        UsageEvent(
            key="layout-1",
            tensor=[1.0, 0.0],
            metadata={"input_tokens": 5},
            compression=record,
        )
    )

    _, stored_metadata = stm.retrieve(key)
    baseline_metadata = record.as_metadata()
    assert stored_metadata["task"] == "layout"
    assert stored_metadata["K"] == baseline_metadata["K"]
    assert stored_metadata["mi_lb"] == pytest.approx(baseline_metadata["mi_lb"])
    assert "idx_cells" in stored_metadata
    assert "layout_signature" in stored_metadata
    assert "canonical_cells" in stored_metadata.get("compression", {}).get(
        "artifacts", {}
    )
    assert "canonical_cells" in stored_metadata.get("artifacts", {})

    layout_signature = stored_metadata.get("layout_signature")
    if layout_signature:
        assert stm.list_by_layout(layout_signature) == [key]


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
        auto_attach_meta_model=False,
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

    strategy = CompressionRatioBudgetStrategy(
        lower_bound=0.6, upper_bound=0.8, step=0.5, min_budget=0.5
    )
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
    assert (
        orchestrator.budget_history[-1].metadata["average_ratio"]
        <= orchestrator.budget_history[-2].metadata["average_ratio"]
    )


def test_meta_model_guides_budget_and_proxy_trials(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)

    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(
            target_budget=2.0,
            policy_name="meta-test",
            budget_step=0.5,
            retention_probe_sample_size=5,
        ),
    )

    assert orchestrator.meta_model is not None
    model_name = getattr(
        orchestrator.meta_model,
        "name",
        orchestrator.meta_model.__class__.__name__,
    )

    record_text = _make_record(
        total_tokens=10, kept_tokens=10, budget=10, field="text", field_mi=0.6
    )
    record_vision = _make_record(
        total_tokens=12, kept_tokens=12, budget=12, field="vision", field_mi=0.9
    )

    orchestrator.log_usage_event(
        UsageEvent(key="meta-1", tensor=[0.1, 0.2], compression=record_text)
    )
    orchestrator.log_usage_event(
        UsageEvent(key="meta-2", tensor=[0.3, 0.4], compression=record_vision)
    )

    tuned_budget = orchestrator.tune_budget()
    assert tuned_budget > 2.0
    assert orchestrator.config.target_budget == pytest.approx(tuned_budget)
    assert orchestrator.budget_history[-1].reason.startswith("meta-model")
    meta_summary = orchestrator.budget_history[-1].metadata["meta_model"]
    assert meta_summary["model"] == model_name
    selected = meta_summary["selected"]
    assert selected["candidate"]["budget"] == pytest.approx(tuned_budget)
    assert selected["score"] > 0

    proxy = orchestrator.run_proxy_trial(
        candidate_budget=1.25, include_adversarial=True, window=2
    )
    assert proxy["candidate"]["budget"] == pytest.approx(1.25)
    assert proxy["meta_model"]["model"] == model_name
    assert proxy["meta_model"]["candidate"]["budget"] == pytest.approx(1.25)
    assert proxy["meta_model"]["score"] == pytest.approx(proxy["score"])
    assert proxy["adversarial_samples"]

    samples = orchestrator.generate_adversarial_samples(limit=1)
    assert samples and samples[0]["key"] in orchestrator.usage_log

    final_record = _make_record(
        total_tokens=8, kept_tokens=4, budget=4, field="text", field_mi=0.2
    )
    final_key = orchestrator.log_usage_event(
        UsageEvent(key="meta-final", tensor=[0.5, 0.6], compression=final_record)
    )

    index_entry = stm.get_index_entry(final_key)
    compression_meta = index_entry["metadata"]["compression"]
    assert compression_meta["policy_metadata"]["policy"] == "meta-test"
    assert (
        compression_meta["policy_metadata"]["decision"]["metadata"]["meta_model"][
            "model"
        ]
        == model_name
    )
    probe_types = {entry["type"] for entry in compression_meta["probe_outcomes"]}
    assert "proxy_trial" in probe_types


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
    assert 0.0 <= reconstruction["mean_quality"] <= 1.0
    assert 0.0 <= reconstruction["mean_retained_ratio"] <= 1.0
    assert reconstruction["mean_mse"] >= 0.0
    assert len(reconstruction.get("details", [])) == 2
    assert probe["meta_model"]["model"] == getattr(
        orchestrator.meta_model,
        "name",
        orchestrator.meta_model.__class__.__name__,
    )


def test_from_components_attaches_default_meta_model(tmp_path) -> None:
    orchestrator = Orchestrator.from_components(
        target_budget=2.0,
        policy_name="component-meta",
        budget_step=0.5,
        retention_probe_sample_size=5,
        storage_dir=tmp_path,
    )

    assert orchestrator.meta_model is not None
    model_name = getattr(
        orchestrator.meta_model,
        "name",
        orchestrator.meta_model.__class__.__name__,
    )

    record = _make_record(
        total_tokens=10,
        kept_tokens=10,
        budget=10,
        field="text",
        field_mi=0.8,
    )
    orchestrator.log_usage_event(
        UsageEvent(key="component-1", tensor=[0.1, 0.2], compression=record)
    )

    proposed = orchestrator.tune_budget()
    assert proposed >= orchestrator.config.budget_step
    decision = orchestrator.budget_history[-1]
    assert decision.reason.startswith("meta-model")
    meta_summary = decision.metadata["meta_model"]
    assert meta_summary["model"] == model_name
    assert meta_summary["selected"]["candidate"]["budget"] == pytest.approx(proposed)
    assert meta_summary["selected"]["score"] > 0


def test_orchestrator_runs_constraints_and_superpositions(tmp_path) -> None:
    storage_config = STMConfig(storage_dir=tmp_path)
    stm = STM(storage_config)
    constraint = FieldActivationConstraint(field="text", min_tokens=1)
    orchestrator = Orchestrator(
        stm=stm,
        config=OrchestratorConfig(
            target_budget=1.0, policy_name="constraint-test", budget_step=0.5
        ),
        constraints=[constraint],
        superposition_channels=("usage",),
    )

    telemetry = CompressionTelemetry(
        selected_indices={"text": [0]},
        selected_scores={"text": [0.5]},
        token_counts={"text": 1},
        budget=1,
        field_budgets={"text": 1},
        allocation_weights={"text": 1.0},
        dropped_indices={"text": []},
        residual_statistics={},
        quantized_embeddings={},
    )
    result = CompressionResult(
        compressed_fields={"text": ["token"]},
        telemetry=telemetry,
        metrics={"ib_proxy": 0.1},
    )
    record = CompressionRecord.from_result(result, bottleneck="ib-test")
    event = UsageEvent(
        key="constraint-event",
        tensor=[0.1, 0.2],
        metadata={},
        compression=record,
    )

    key = orchestrator.log_usage_event(event)
    entry = stm.get_index_entry(key)
    assert "constraints" in entry["metadata"]
    vector, info = stm.read_superposition("usage")
    assert vector
    assert info["channel"] == "usage"
