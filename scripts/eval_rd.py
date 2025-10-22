"""Token budget sweep for the NDEncoderDecoder scaffold."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
        "encoder_latency": 0.0,
        "encoder_flops": 0.0,
        "registration_pre": 0.0,
        "registration_post": 0.0,
    }
    steps = 0
    label_counts: Counter[int] = Counter()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            targets = batch.get("targets")
            if isinstance(targets, torch.Tensor):
                batch["targets"] = targets.to(device)
            start_time = time.perf_counter()
            logits, logs = model(batch, token_budget=budget)
            latency = time.perf_counter() - start_time
            batch_targets = logs.get("targets")
            if batch_targets is None or not isinstance(batch_targets, torch.Tensor) or batch_targets.numel() == 0:
                continue
            label_counts.update(int(value) for value in batch_targets.tolist() if int(value) >= 0)
            ce = F.cross_entropy(logits, batch_targets, ignore_index=-100)
            totals["cross_entropy"] += float(ce.detach().cpu())
            totals["accuracy"] += average_accuracy(logits, batch_targets)
            totals["mi_lb"] += float(logs.get("mi_lb", 0.0))
            tokens_tensor = logs.get("tokens_selected")
            if isinstance(tokens_tensor, torch.Tensor) and tokens_tensor.numel():
                totals["tokens"] += float(tokens_tensor.float().mean().detach().cpu())
            totals["encoder_latency"] += latency
            totals["encoder_flops"] += _estimate_flops_from_logs(logs, getattr(model, "hidden_dim", 1))
            pre_dist, post_dist = _registration_metrics_from_logs(logs)
            totals["registration_pre"] += pre_dist
            totals["registration_post"] += post_dist
            steps += 1
    if steps:
        for key in (
            "cross_entropy",
            "accuracy",
            "mi_lb",
            "tokens",
            "encoder_latency",
            "encoder_flops",
            "registration_pre",
            "registration_post",
        ):
            totals[key] /= steps
    label_entropy = _entropy(label_counts)
    totals["label_entropy"] = label_entropy
    conditional_entropy = max(0.0, label_entropy - totals.get("mi_lb", 0.0))
    totals["conditional_entropy"] = conditional_entropy
    totals["fano_error_bound"] = _fano_lower_bound(conditional_entropy, len(label_counts))
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


def _estimate_flops_from_logs(logs: Mapping[str, Any], hidden_dim: int) -> float:
    tokens_available = logs.get("tokens_available")
    if isinstance(tokens_available, torch.Tensor) and tokens_available.numel():
        available = float(tokens_available.float().mean().detach().cpu())
    else:
        available = 0.0
    width = float(hidden_dim or 1)
    return available * width * width


def _registration_metrics_from_logs(logs: Mapping[str, Any]) -> tuple[float, float]:
    cell_metadata = logs.get("cell_metadata")
    selected_metadata = logs.get("selected_metadata")
    pre = _metadata_dispersion(cell_metadata)
    post = _metadata_dispersion(selected_metadata) if selected_metadata else pre
    return pre, post


def _metadata_dispersion(metadata: Any) -> float:
    if not metadata:
        return 0.0
    coords: List[tuple[float, float]] = []
    for doc in metadata:
        for entry in doc:
            if not isinstance(entry, Mapping):
                continue
            coord = entry.get("coords")
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                coords.append((float(coord[0]), float(coord[1])))
    if not coords:
        return 0.0
    mean_y = sum(y for y, _ in coords) / len(coords)
    mean_x = sum(x for _, x in coords) / len(coords)
    variance = sum((y - mean_y) ** 2 + (x - mean_x) ** 2 for y, x in coords) / len(coords)
    return math.sqrt(max(variance, 0.0))


def _entropy(counts: Mapping[int, int]) -> float:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def _fano_lower_bound(conditional_entropy: float, num_labels: int) -> float:
    if num_labels <= 1:
        return 0.0
    denom = math.log2(float(num_labels))
    if denom == 0:
        return 0.0
    return max(0.0, (conditional_entropy - 1.0) / denom)
