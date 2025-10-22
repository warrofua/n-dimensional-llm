"""Token budget sweep for the NDEncoderDecoder scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import average_accuracy, build_invoice_dataloader, build_invoice_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="Token budgets to evaluate",
    )
    parser.add_argument("--dataset-size", type=int, default=24, help="Synthetic dataset size")
    parser.add_argument("--threshold", type=float, default=500.0, help="Invoice threshold")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path produced by scripts/train.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store the evaluation report",
    )
    return parser


def _evaluate_budget(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    budget: int,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    totals = {
        "budget": float(budget),
        "cross_entropy": 0.0,
        "accuracy": 0.0,
        "mi_lb": 0.0,
        "tokens": 0.0,
    }
    steps = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            targets = batch.get("targets")
            if isinstance(targets, torch.Tensor):
                batch["targets"] = targets.to(device)
            logits, logs = model(batch, token_budget=budget)
            batch_targets = logs.get("targets")
            if batch_targets is None or not isinstance(batch_targets, torch.Tensor) or batch_targets.numel() == 0:
                continue
            ce = F.cross_entropy(logits, batch_targets, ignore_index=-100)
            totals["cross_entropy"] += float(ce.detach().cpu())
            totals["accuracy"] += average_accuracy(logits, batch_targets)
            totals["mi_lb"] += float(logs.get("mi_lb", 0.0))
            tokens_tensor = logs.get("tokens_selected")
            if isinstance(tokens_tensor, torch.Tensor) and tokens_tensor.numel():
                totals["tokens"] += float(tokens_tensor.float().mean().detach().cpu())
            steps += 1
    if steps:
        for key in ("cross_entropy", "accuracy", "mi_lb", "tokens"):
            totals[key] /= steps
    return totals


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist")
    state_dict = torch.load(path, map_location=next(model.parameters()).device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    model = build_invoice_model(hidden_dim=args.hidden_dim)
    device = torch.device("cpu")
    model.to(device)
    if args.checkpoint is not None:
        _load_checkpoint(model, args.checkpoint)

    dataloader = build_invoice_dataloader(
        dataset_size=args.dataset_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    budget_reports: List[Dict[str, float]] = []
    for budget in args.budgets:
        metrics = _evaluate_budget(model, dataloader, budget=budget)
        metrics["budget"] = float(budget)
        budget_reports.append(metrics)
        print(
            f"budget={budget}: ce={metrics['cross_entropy']:.4f} mi_lb={metrics['mi_lb']:.4f} "
            f"acc={metrics['accuracy']:.3f} tokens={metrics['tokens']:.3f}"
        )

    report = {
        "dataset": "synthetic-invoice",
        "dataset_size": args.dataset_size,
        "threshold": args.threshold,
        "budgets": budget_reports,
    }
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
