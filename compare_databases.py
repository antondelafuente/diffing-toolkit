#!/usr/bin/env python3
"""Compare the two databases to verify they have the same content for overlapping features."""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import json
from pathlib import Path
import random

# Load active_positions from both databases
db_full = Path('/workspace/diffing-toolkit/efficient_feature_db_full')
db_safe = Path('/workspace/diffing-toolkit/efficient_feature_db_memory_safe')

print("Loading active positions from both databases...")

with open(db_full / 'active_positions.json', 'r') as f:
    positions_full = json.load(f)

with open(db_safe / 'active_positions.json', 'r') as f:
    positions_safe = json.load(f)

print(f"Full DB has {len(positions_full)} features")
print(f"Memory-safe DB has {len(positions_safe)} features")

# Find overlapping features
overlapping = set(positions_full.keys()) & set(positions_safe.keys())
print(f"Overlapping features: {len(overlapping)}")

# Random sample some features to check
sample_size = min(10, len(overlapping))
sampled_features = random.sample(list(overlapping), sample_size)

print(f"\nChecking {sample_size} random features for consistency:")
print("="*60)

matches = 0
mismatches = 0

for feat_id in sampled_features:
    full_examples = positions_full[feat_id]
    safe_examples = positions_safe[feat_id]
    
    # Both should have exactly 20 examples (or same number)
    if len(full_examples) != len(safe_examples):
        print(f"Feature {feat_id}: Different number of examples! Full={len(full_examples)}, Safe={len(safe_examples)}")
        mismatches += 1
        continue
    
    # Check if the examples are the same (comparing positions)
    full_set = set(tuple(ex) for ex in full_examples)
    safe_set = set(tuple(ex) for ex in safe_examples)
    
    if full_set == safe_set:
        matches += 1
        print(f"Feature {feat_id}: ✓ MATCH - {len(full_examples)} examples identical")
    else:
        mismatches += 1
        overlap = full_set & safe_set
        print(f"Feature {feat_id}: ✗ MISMATCH - {len(overlap)}/{len(full_examples)} examples overlap")
        
        # Show first example from each for comparison
        print(f"  Full DB first: {full_examples[0]}")
        print(f"  Safe DB first: {safe_examples[0]}")

print("\n" + "="*60)
print(f"Summary: {matches} matches, {mismatches} mismatches out of {sample_size} checked")

# Check specific important features
print("\nChecking specific features:")
important_features = ['11163', '0', '100', '1000', '5000', '10000']

for feat_id in important_features:
    in_full = feat_id in positions_full
    in_safe = feat_id in positions_safe
    
    if in_full and in_safe:
        full_ex = positions_full[feat_id]
        safe_ex = positions_safe[feat_id]
        if set(tuple(ex) for ex in full_ex) == set(tuple(ex) for ex in safe_ex):
            print(f"Feature {feat_id}: ✓ Present in both, examples match")
        else:
            print(f"Feature {feat_id}: Present in both but examples differ")
    elif in_safe and not in_full:
        print(f"Feature {feat_id}: ✓ Only in memory-safe (as expected - full was incomplete)")
    elif in_full and not in_safe:
        print(f"Feature {feat_id}: ✗ Missing from memory-safe (unexpected!)")
    else:
        print(f"Feature {feat_id}: Not in either database")