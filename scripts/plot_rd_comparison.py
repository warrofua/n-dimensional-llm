#!/usr/bin/env python3
"""Generate ND vs text-only comparison plot from rd_audit results."""

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    exit(1)


def plot_rd_comparison(results_path: Path, output_path: Path):
    """Generate comparison plot from rd_audit JSON results."""

    with open(results_path) as f:
        data = json.load(f)

    nd_results = data["modes"]["nd"]["results"]
    text_results = data["modes"]["text"]["results"]

    # Extract budgets and distortions
    budgets_nd = [r["budget"] for r in nd_results]
    distortions_nd = [r["distortion"] for r in nd_results]

    budgets_text = [r["budget"] for r in text_results]
    distortions_text = [r["distortion"] for r in text_results]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot both curves
    plt.plot(
        budgets_nd,
        distortions_nd,
        "o-",
        linewidth=2,
        markersize=8,
        label="N-D (multi-field)",
        color="#2563eb",
    )
    plt.plot(
        budgets_text,
        distortions_text,
        "s--",
        linewidth=2,
        markersize=8,
        label="Text-only",
        color="#dc2626",
    )

    # Formatting
    plt.xlabel("Token Budget (K)", fontsize=12, fontweight="bold")
    plt.ylabel("Distortion (Error Rate)", fontsize=12, fontweight="bold")
    plt.title(
        f'Rate-Distortion: N-D vs Text-Only on {data["dataset"]}\n(n={data["dataset_size"]})',
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="best")

    # Set y-axis to start at 0
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dataset: {data['dataset']}")
    print(f"Documents: {data['dataset_size']}")
    print(f"\n{'Budget':<10} {'N-D Dist.':<15} {'Text Dist.':<15} {'N-D Better?'}")
    print("-" * 60)

    for nd_r, text_r in zip(nd_results, text_results):
        budget = nd_r["budget"]
        nd_dist = nd_r["distortion"]
        text_dist = text_r["distortion"]
        better = "✓" if nd_dist <= text_dist else "✗"
        print(f"{budget:<10} {nd_dist:<15.4f} {text_dist:<15.4f} {better}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ND vs text-only R-D curves")
    parser.add_argument(
        "--input",
        type=Path,
        default="runs/cord_rd_sample.json",
        help="Path to rd_audit JSON results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="docs/figs/cord_rd_comparison.png",
        help="Path to save output plot",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run rd_audit.py first to generate results.")
        exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_rd_comparison(args.input, args.output)
