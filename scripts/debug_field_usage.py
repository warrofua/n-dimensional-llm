#!/usr/bin/env python3
"""Debug script to see which fields are being selected in compression."""

from benchmarks.cord import (
    build_cord_encoders,
    build_cord_registry,
    cord_fields,
    load_cord_dataset,
)
from nd_llm.bottleneck import IBottleneck
from nd_llm.utils import build_mi_proxy_context

# Load one document
docs = load_cord_dataset(data_root="datasets/CORD 2", split="train", limit=5)

registry = build_cord_registry()
build_cord_encoders(registry)

print("=" * 80)
print("FIELD SELECTION ANALYSIS")
print("=" * 80)

for budget in [4, 16, 64]:
    print(f"\n{'='*80}")
    print(f"BUDGET = {budget}")
    print(f"{'='*80}")

    bottleneck = IBottleneck(target_budget=budget)

    for doc_idx, doc in enumerate(docs[:3]):
        fields = cord_fields(doc)

        # Count tokens per field BEFORE compression
        print(f"\nDocument {doc_idx}: {doc.get('doc_id')}")
        print("  Input tokens per field:")
        for field_name, entries in fields.items():
            print(f"    {field_name}: {len(entries)}")

        mi_proxy, mi_context = build_mi_proxy_context(fields, registry.encoders)
        result = bottleneck.compress(
            fields,
            encoders=registry.encoders,
            registry=registry,  # Pass registry so budget allocator can see field metadata!
            context=mi_context,
            mi_proxy=mi_proxy,
        )

        # Count tokens per field AFTER compression
        print("  Kept tokens per field:")
        total_kept = 0
        for field_name, indices in result.telemetry.selected_indices.items():
            num_kept = len(indices)
            total_kept += num_kept
            pct = (
                100 * num_kept / len(fields.get(field_name, [1]))
                if fields.get(field_name)
                else 0
            )
            print(
                f"    {field_name}: {num_kept}/{len(fields.get(field_name, []))} ({pct:.1f}%)"
            )

        print(f"  Total kept: {total_kept}/{budget}")
