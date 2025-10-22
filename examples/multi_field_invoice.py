"""End-to-end multi-field invoice demo using the ND-LLM stubs."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory
from typing import Any, Dict, Mapping, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nd_llm.bottleneck import IBottleneck
from nd_llm.orchestration import CompressionRecord, Orchestrator, UsageEvent
from nd_llm.stm import STM
from nd_llm.utils import OrchestratorConfig, STMConfig

from benchmarks.synthetic import (
    build_invoice_encoders,
    build_invoice_registry,
    invoice_fields,
    synthetic_invoice,
)


def run_demo(
    storage_dir: Path | None = None,
    *,
    budget: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the multi-field invoice demo and return a summary payload."""

    registry = build_invoice_registry()
    encoders = build_invoice_encoders(registry)
    invoice = synthetic_invoice(doc_id=1, num_lines=6, seed=seed)
    fields = invoice_fields(invoice)

    bottleneck = IBottleneck(target_budget=budget)
    compression = bottleneck.compress(fields, encoders=registry.encoders)
    reconstructed = bottleneck.decompress(compression)

    def _execute(storage_path: Path) -> Dict[str, Any]:
        stm = STM(STMConfig(storage_dir=storage_path))
        orchestrator = Orchestrator(
            stm=stm,
            config=OrchestratorConfig(
                target_budget=float(budget),
                policy_name="example-invoice",
                budget_step=1.0,
                retention_probe_sample_size=5,
            ),
        )

        scores_vector = _scores_to_tensor(compression.telemetry.selected_scores)
        kept_tokens = sum(len(v) for v in compression.telemetry.selected_indices.values())
        stm_key = orchestrator.log_usage_event(
            UsageEvent(
                tensor=[scores_vector],
                metadata={
                    "doc_id": invoice["doc_id"],
                    "vendor": invoice["vendor"],
                    "kept_tokens": kept_tokens,
                },
                compression=CompressionRecord.from_result(
                    compression,
                    bottleneck=bottleneck,
                ),
            )
        )

        stored_tensor, stored_metadata = stm.retrieve(stm_key)
        if hasattr(stored_tensor, "tolist"):
            stored_tensor = stored_tensor.tolist()
        budget_probe = orchestrator.budget_sweep()
        retention_probe = orchestrator.run_retention_probe()

        return {
            "registry": registry,
            "invoice": invoice,
            "fields": fields,
            "compression": compression,
            "reconstructed": reconstructed,
            "stm_key": stm_key,
            "stm_entry": {"tensor": stored_tensor, "metadata": stored_metadata},
            "budget_probe": budget_probe,
            "retention_probe": retention_probe,
            "usage_log": orchestrator.usage_log,
        }

    if storage_dir is None:
        with TemporaryDirectory(prefix="ndllm-example-") as tmp:
            return _execute(Path(tmp))
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    return _execute(storage_path)


def _scores_to_tensor(scores: Mapping[str, Sequence[float]]) -> Sequence[float]:
    vector: list[float] = []
    for field in sorted(scores):
        vector.extend(float(value) for value in scores[field])
    if not vector:
        vector.append(0.0)
    return vector


def main() -> None:
    """CLI entry point that prints demo artefacts."""

    summary = run_demo()
    print("Compression metrics:")
    pprint(summary["compression"].metrics)
    print("\nSTM metadata:")
    pprint(summary["stm_entry"]["metadata"])
    print("\nBudget probe:")
    pprint(summary["budget_probe"])
    print("\nRetention probe:")
    pprint(summary["retention_probe"])


if __name__ == "__main__":
    main()
