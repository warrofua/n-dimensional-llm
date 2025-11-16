#!/usr/bin/env python3
"""Rate–distortion and Fano audit utility for CORD receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmarks.cord import (
    build_cord_encoders,
    build_cord_registry,
    cord_fields,
    cord_high_total_label,
    load_cord_dataset,
)
from benchmarks import doc_understanding as doc_bench
from nd_llm.bottleneck import IBottleneck
from nd_llm.utils import build_mi_proxy_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep token budgets for the CORD dataset in both N-D and text-only "
            "configurations, emitting rate–distortion and Fano summaries."
        )
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="Token budgets to evaluate (default: 4 8 12).",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=8,
        help="Number of documents to sample (default: 8).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=250_000.0,
        help="Receipt total threshold for high-value classification.",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use the bundled JSON sample instead of downloading from Hugging Face.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split passed to Hugging Face (default: train).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional path to local CORD dataset directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for Hugging Face datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report (defaults to stdout).",
    )
    return parser.parse_args()


def label_entropy_bits(labels: Sequence[bool]) -> Tuple[float, int]:
    counts: Dict[bool, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values()) or 1
    entropy = 0.0
    for value in counts.values():
        prob = value / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy, len(counts)


def evaluate_mode(
    *,
    mode: str,
    budgets: Sequence[int],
    dataset: Sequence[Mapping[str, Any]],
    registry_encoders: Mapping[str, Any],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], float]:
    predict_fn = doc_bench._cord_prediction_factory(threshold)  # type: ignore[attr-defined]
    budgets_summary: List[Dict[str, Any]] = []
    total_mi = 0.0
    total_mi_count = 0

    for budget in budgets:
        bottleneck = IBottleneck(target_budget=int(budget))
        correct = 0
        kept_totals = 0
        info_bound_sum = 0.0
        rate_dist_sum = 0.0
        mi_sum = 0.0
        mi_count = 0
        doc_count = 0

        for document in dataset:
            fields = cord_fields(document)
            if mode == "text":
                fields = {"text": list(fields.get("text", []))}
            mi_proxy, mi_context = build_mi_proxy_context(
                fields,
                registry_encoders,
                preferred_fields=tuple(fields),
            )
            result = bottleneck.compress(
                fields,
                encoders=registry_encoders,
                context=mi_context,
                mi_proxy=mi_proxy,
            )
            label = cord_high_total_label(document, threshold=threshold)
            prediction = predict_fn(result, document)
            if prediction == label:
                correct += 1
            kept = sum(
                len(indices) for indices in result.telemetry.selected_indices.values()
            )
            kept_totals += kept
            metrics = result.metrics
            info_bound_sum += float(metrics.get("information_bound", 0.0) or 0.0)
            rate_dist_sum += float(metrics.get("rate_distortion", 0.0) or 0.0)
            mi_value = metrics.get("mi_lower_bound")
            if mi_value is not None:
                mi_sum += float(mi_value)
                mi_count += 1
            doc_count += 1

        accuracy = float(correct) / float(doc_count or 1)
        avg_kept = float(kept_totals) / float(doc_count or 1)
        mean_info_bound = info_bound_sum / float(doc_count or 1)
        mean_rate_dist = rate_dist_sum / float(doc_count or 1)
        mean_mi = mi_sum / float(mi_count or 1) if mi_count else 0.0
        total_mi += mi_sum
        total_mi_count += mi_count

        budgets_summary.append(
            {
                "budget": int(budget),
                "accuracy": accuracy,
                "distortion": 1.0 - accuracy,
                "average_kept_tokens": avg_kept,
                "mean_information_bound": mean_info_bound,
                "mean_rate_distortion": mean_rate_dist,
                "mean_mi_lower_bound_nats": mean_mi,
                "evaluated_documents": doc_count,
            }
        )

    overall_mi = total_mi / float(total_mi_count or 1) if total_mi_count else 0.0
    return budgets_summary, overall_mi


def fano_error_bound(
    *,
    label_entropy_bits: float,
    num_labels: int,
    mi_lower_bound_nats: float,
) -> float:
    if num_labels <= 1:
        return 0.0
    mi_bits = mi_lower_bound_nats / math.log(2) if mi_lower_bound_nats else 0.0
    numerator = label_entropy_bits - mi_bits - 1.0
    denom = math.log2(num_labels)
    if denom <= 0:
        return 0.0
    return max(0.0, numerator / denom)


def main() -> None:
    args = parse_args()
    dataset = load_cord_dataset(
        split=args.split,
        limit=args.dataset_size,
        use_sample=args.use_sample,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
    )
    if not dataset:
        raise RuntimeError("No documents available for the requested configuration.")

    registry = build_cord_registry()
    build_cord_encoders(registry)
    labels = [cord_high_total_label(doc, threshold=args.threshold) for doc in dataset]
    entropy_bits, num_labels = label_entropy_bits(labels)

    nd_summary, nd_mi = evaluate_mode(
        mode="nd",
        budgets=args.budgets,
        dataset=dataset,
        registry_encoders=registry.encoders,
        threshold=args.threshold,
    )
    text_summary, text_mi = evaluate_mode(
        mode="text",
        budgets=args.budgets,
        dataset=dataset,
        registry_encoders=registry.encoders,
        threshold=args.threshold,
    )

    report = {
        "dataset": "CORD-v2",
        "split": args.split,
        "dataset_size": len(dataset),
        "use_sample": bool(args.use_sample),
        "threshold": float(args.threshold),
        "label_entropy_bits": entropy_bits,
        "budgets": [int(b) for b in args.budgets],
        "modes": {
            "nd": {
                "mean_mi_lower_bound_nats": nd_mi,
                "fano_error_lower_bound": fano_error_bound(
                    label_entropy_bits=entropy_bits,
                    num_labels=num_labels,
                    mi_lower_bound_nats=nd_mi,
                ),
                "results": nd_summary,
            },
            "text": {
                "mean_mi_lower_bound_nats": text_mi,
                "fano_error_lower_bound": fano_error_bound(
                    label_entropy_bits=entropy_bits,
                    num_labels=num_labels,
                    mi_lower_bound_nats=text_mi,
                ),
                "results": text_summary,
            },
        },
    }

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
