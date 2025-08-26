#!/usr/bin/env python3
"""Compare feature 11163 examples from the new database."""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import json
from pathlib import Path
from src.utils.max_act_store import MaxActStore
from transformers import AutoTokenizer

# Load the new database
db_path = Path('/workspace/diffing-toolkit/efficient_feature_db_memory_safe')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
max_store = MaxActStore(db_path / "examples.db", tokenizer=tokenizer)

# Get examples for feature 11163
feature_id = 11163
examples = max_store.get_top_examples(feature_id, k=20)

print("="*60)
print("Feature 11163 examples from new database:")
print("="*60)

if examples and 'examples' in examples:
    for i, ex in enumerate(examples['examples'][:5], 1):  # Show top 5
        print(f"\n{i}. Score: {ex['max_act']:.6f}")
        print(f"   Tokens: {ex['tokens'][:100]}...")  # First 100 chars of token string
        
print("\n" + "="*60)
print("Comparing with earlier analysis:")
print("="*60)

# Load earlier analysis
with open('/workspace/diffing-toolkit/feature_11163_analysis.json', 'r') as f:
    earlier = json.load(f)

print(f"Earlier max activation: {earlier['max_activation']:.6f}")
print(f"Earlier had {len(earlier['top_examples'])} examples")
print(f"\nTop 3 earlier activations:")
for i, ex in enumerate(earlier['top_examples'][:3], 1):
    print(f"{i}. Activation: {ex['activation']:.6f} (position {ex['position']})")
    print(f"   Context: {ex['context'][:100]}...")

# Load active positions to get more detail
with open(db_path / 'active_positions.json', 'r') as f:
    positions = json.load(f)

if '11163' in positions:
    print(f"\nNew database has {len(positions['11163'])} examples for feature 11163")
    print("Sequence indices and positions:", positions['11163'][:5])