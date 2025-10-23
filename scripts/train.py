# mypy: ignore-errors
"""Command-line training harness for the NDEncoderDecoder scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from nd_llm.bottleneck import CompressionTelemetry
from .common import (
    average_accuracy,
    build_invoice_dataloader,
    build_invoice_model,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--dataset-size", type=int, default=24, help="Synthetic dataset size")
    parser.add_argument("--threshold", type=float, default=500.0, help="Invoice amount threshold")
    parser.add_argument("--budget", type=int, default=8, help="Token budget for the bottleneck")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Rate penalty coefficient")
    parser.add_argument("--beta", type=float, default=0.1, help="Mutual-information weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write training history as JSON",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default=None,
        help="Optional bottleneck scorer configuration (string or JSON)",
    )
    return parser


def _compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logs: Mapping[str, Any],
    *,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(logits, targets, ignore_index=-100)
    tokens_tensor = logs.get("tokens_selected")
    if isinstance(tokens_tensor, torch.Tensor) and tokens_tensor.numel():
        rate_pen = float(alpha) * tokens_tensor.float().mean()
    else:
        rate_pen = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    if not isinstance(rate_pen, torch.Tensor):
        rate_pen = torch.tensor(rate_pen, dtype=logits.dtype, device=logits.device)
    mi_tensor = logs.get("mi_lb_tensor")
    if not isinstance(mi_tensor, torch.Tensor):
        mi_tensor = torch.tensor(float(logs.get("mi_lb", 0.0)), dtype=logits.dtype, device=logits.device)
    mi_term = -float(beta) * mi_tensor
    loss = ce + rate_pen + mi_term
    metrics = {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "rate": float(rate_pen.detach().cpu()),
        "mi": float(mi_term.detach().cpu()),
        "mi_lb": float(mi_tensor.detach().cpu()),
    }
    return loss, metrics


def _train_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    budget: int,
    alpha: float,
    beta: float,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.train()
    totals: Dict[str, float] = {
        "loss": 0.0,
        "ce": 0.0,
        "rate": 0.0,
        "mi": 0.0,
        "mi_lb": 0.0,
        "accuracy": 0.0,
        "tokens": 0.0,
    }
    field_budget_totals: Dict[str, int] = {}
    allocation_weight_totals: Dict[str, float] = {}
    steps = 0
    for batch in dataloader:
        batch_targets = batch.get("targets")
        if isinstance(batch_targets, torch.Tensor):
            batch["targets"] = batch_targets.to(device)
        optimizer.zero_grad()
        logits, logs = model(batch, token_budget=budget)
        targets = logs.get("targets")
        if targets is None or not isinstance(targets, torch.Tensor) or targets.numel() == 0:
            continue
        loss, parts = _compute_loss(logits, targets, logs, alpha=alpha, beta=beta)
        loss.backward()
        optimizer.step()
        for key in ("loss", "ce", "rate", "mi", "mi_lb"):
            totals[key] += parts[key]
        totals["accuracy"] += average_accuracy(logits.detach(), targets.detach())
        tokens_tensor = logs.get("tokens_selected")
        if isinstance(tokens_tensor, torch.Tensor) and tokens_tensor.numel():
            totals["tokens"] += float(tokens_tensor.float().mean().detach().cpu())
        telemetry = logs.get("compression_telemetry")
        if isinstance(telemetry, CompressionTelemetry):
            for field, value in telemetry.field_budgets.items():
                key = str(field)
                field_budget_totals[key] = field_budget_totals.get(key, 0) + int(value)
            for field, value in telemetry.allocation_weights.items():
                key = str(field)
                current_weight = allocation_weight_totals.get(key, 0.0)
                allocation_weight_totals[key] = current_weight + float(value)
        steps += 1
    if steps == 0:
        result: Dict[str, Any] = {key: 0.0 for key in totals}
        result["field_budgets"] = {}
        result["allocation_weights"] = {}
        return result
    averages: Dict[str, Any] = {key: value / steps for key, value in totals.items()}
    averages["field_budgets"] = {
        field: float(total) / steps for field, total in field_budget_totals.items()
    }
    averages["allocation_weights"] = {
        field: total / steps for field, total in allocation_weight_totals.items()
    }
    return averages


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    if args.scorer is not None:
        try:
            scorer_config: Optional[Any] = json.loads(args.scorer)
        except json.JSONDecodeError:
            scorer_config = args.scorer
    else:
        scorer_config = None

    model = build_invoice_model(hidden_dim=args.hidden_dim, scorer=scorer_config)
    device = torch.device("cpu")
    model.to(device)
    dataloader = build_invoice_dataloader(
        dataset_size=args.dataset_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = _train_epoch(
            model,
            dataloader,
            optimizer,
            budget=args.budget,
            alpha=args.alpha,
            beta=args.beta,
        )
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        budget_summary = epoch_metrics.get("field_budgets", {})
        weight_summary = epoch_metrics.get("allocation_weights", {})
        budget_text = ""
        weight_text = ""
        if isinstance(budget_summary, Mapping) and budget_summary:
            parts = [f"{field}:{value:.2f}" for field, value in sorted(budget_summary.items())]
            budget_text = f" budgets=[{', '.join(parts)}]"
        if isinstance(weight_summary, Mapping) and weight_summary:
            parts = [f"{field}:{value:.2f}" for field, value in sorted(weight_summary.items())]
            weight_text = f" alloc=[{', '.join(parts)}]"
        print(
            f"epoch {epoch}: loss={epoch_metrics['loss']:.4f} ce={epoch_metrics['ce']:.4f} "
            f"rate={epoch_metrics['rate']:.4f} mi_lb={epoch_metrics['mi_lb']:.4f} "
            f"acc={epoch_metrics['accuracy']:.3f} tokens={epoch_metrics['tokens']:.3f}" \
            f"{budget_text}{weight_text}"
        )

    result = {"history": history}
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
