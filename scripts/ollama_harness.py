#!/usr/bin/env python3
"""Run compressed field prompts through a local Ollama model."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from benchmarks.cord import (
    build_cord_encoders,
    build_cord_registry,
    cord_fields,
    cord_high_total_label,
    cord_total_amount,
    load_cord_dataset,
)
from benchmarks.chartqa import (
    build_chartqa_encoders,
    build_chartqa_registry,
    chartqa_answer,
    chartqa_fields,
    load_chartqa_dataset,
)
from nd_llm.bottleneck import IBottleneck
from nd_llm.utils import build_mi_proxy_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress structured fields then prompt a local Ollama model."
    )
    parser.add_argument(
        "--dataset",
        choices=("cord", "chartqa"),
        default="cord",
        help="Dataset harness to use (default: cord).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=6,
        help="Target bottleneck budget (default: 6).",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1,
        help="Number of documents to load before sampling one (default: 1).",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use bundled JSONL samples instead of the full dataset.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional local path to the dataset (e.g. datasets/).",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model name (default: llama3.1:8b).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=250_000.0,
        help="CORD high-value threshold (default: 250000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt JSON without contacting the Ollama server.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Base URL for the Ollama server (default: http://127.0.0.1:11434).",
    )
    return parser.parse_args()


def load_document(args: argparse.Namespace) -> tuple[Mapping[str, Any], str]:
    if args.dataset == "cord":
        docs = load_cord_dataset(
            split="train",
            limit=max(args.dataset_size, 1),
            use_sample=args.use_sample,
            data_root=args.data_root,
        )
        if not docs:
            raise RuntimeError("No CORD documents available for the requested configuration.")
        return docs[0], "cord"
    docs = load_chartqa_dataset(
        split="test",
        limit=max(args.dataset_size, 1),
        use_sample=args.use_sample,
        cache_dir=args.data_root,
    )
    if not docs:
        raise RuntimeError("No ChartQA documents available for the requested configuration.")
    return docs[0], "chartqa"


def compress_document(document: Mapping[str, Any], dataset: str, budget: int) -> tuple[Any, Mapping[str, Any]]:
    if dataset == "cord":
        registry = build_cord_registry()
        build_cord_encoders(registry)
        fields = cord_fields(document)
    else:
        registry = build_chartqa_registry()
        build_chartqa_encoders(registry)
        fields = chartqa_fields(document)
    mi_proxy, mi_context = build_mi_proxy_context(fields, registry.encoders, preferred_fields=tuple(fields))
    bottleneck = IBottleneck(target_budget=int(budget))
    result = bottleneck.compress(fields, encoders=registry.encoders, context=mi_context, mi_proxy=mi_proxy)
    return result, fields


def build_prompt(
    document: Mapping[str, Any],
    result: Any,
    dataset: str,
    *,
    threshold: float,
) -> str:
    if dataset == "cord":
        tokens = _collect_tokens(result.compressed_fields.get("text", []))
        total = cord_total_amount(document)
        doc_id = document.get("doc_id")
        return textwrap.dedent(
            f"""
            You are auditing a receipt (doc_id={doc_id}). Selected text tokens from the variable-rate encoder:

            {tokens or '[no tokens retained]'}

            The goal is to decide whether the receipt should be flagged as HIGH VALUE (total >= {threshold:,.0f}).
            Respond with a single line explaining whether it should be flagged and cite the evidence.
            """
        ).strip()

    question = str(document.get("question", ""))
    chart_rows = document.get("chart") or []
    kept_rows = [
        f"{row.get('label')}: {row.get('value')}"
        for row in result.compressed_fields.get("chart", [])
        if isinstance(row, Mapping)
    ]
    chart_summary = kept_rows or [f"{row.get('label')}: {row.get('value')}" for row in chart_rows[:4]]
    return textwrap.dedent(
        f"""
        You are answering a question about a chart. Selected chart entries:

        - """[1:]
        + "\n- ".join(chart_summary)
        + textwrap.dedent(
            f"""

            Question: {question}

            Provide the answer and cite the most relevant entries.
            """
        )
    ).strip()


def _collect_tokens(entries: Sequence[Any]) -> str:
    tokens: List[str] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            text = entry.get("text")
            if text:
                tokens.append(str(text))
        else:
            tokens.append(str(entry))
    if not tokens:
        return ""
    return " / ".join(tokens[:30])


def call_ollama(model: str, prompt: str, url: str) -> Dict[str, Any]:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            return {"response": data.get("response", ""), "raw": data}
    except urllib.error.URLError as exc:
        return {"error": f"Failed to reach Ollama server: {exc}"}


def main() -> None:
    args = parse_args()
    document, dataset = load_document(args)
    result, _ = compress_document(document, dataset, args.budget)
    prompt = build_prompt(document, result, dataset, threshold=args.threshold)
    summary: Dict[str, Any] = {
        "dataset": dataset,
        "doc_id": document.get("doc_id"),
        "model": args.model,
        "budget": int(args.budget),
        "prompt": prompt,
    }
    if not args.dry_run:
        summary["ollama"] = call_ollama(args.model, prompt, args.ollama_url)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
