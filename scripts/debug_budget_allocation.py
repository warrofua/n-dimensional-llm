#!/usr/bin/env python3
"""Debug script to see how budget allocation weights are calculated."""

from benchmarks.cord import build_cord_registry

registry = build_cord_registry()

print("=" * 80)
print("CORD REGISTRY FIELD METADATA")
print("=" * 80)

for field_name, field_spec in registry.fields.items():
    print(f"\nField: {field_name}")
    print(f"  Keys: {field_spec.keys}")
    print(f"  Salience: {field_spec.salience}")

    # Calculate weight using RegistryAwareBudgetAllocator logic
    key_weight = 0.3  # default
    salience_bonus = 1.5  # default

    weight = 1.0 + key_weight * len(field_spec.keys)
    if field_spec.salience:
        weight *= salience_bonus

    print(f"  Calculated weight: {weight:.2f}")
    print(
        f"    = 1.0 + {key_weight} * {len(field_spec.keys)} = {1.0 + key_weight * len(field_spec.keys):.2f}"
    )
    if field_spec.salience:
        print(f"    * {salience_bonus} (salience bonus) = {weight:.2f}")

print("\n" + "=" * 80)
print("WEIGHT DISTRIBUTION")
print("=" * 80)

# Simulate allocation
weights = {}
for field_name, field_spec in registry.fields.items():
    weight = 1.0 + 0.3 * len(field_spec.keys)
    if field_spec.salience:
        weight *= 1.5
    weights[field_name] = weight

total_weight = sum(weights.values())
print(f"\nTotal weight: {total_weight:.2f}")
print("\nPer-field budget share:")
for field_name, weight in weights.items():
    share = weight / total_weight * 100
    print(f"  {field_name}: {share:.1f}%")
