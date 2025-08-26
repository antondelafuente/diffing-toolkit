#!/usr/bin/env python3
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

# Get examples
feature_id = 11163
try:
    # Try different methods
    print(f"Getting examples for feature {feature_id}...")
    
    # Method 1: get_top_examples with various parameters
    examples = max_store.get_top_examples(feature_id, k=20)
    print(f"Type of examples: {type(examples)}")
    
    if examples is not None:
        print(f"Got examples: {examples}")
        
except Exception as e:
    print(f"Error getting examples: {e}")
    
# Also check the active_positions file directly
with open(db_path / 'active_positions.json', 'r') as f:
    positions = json.load(f)
    
if '11163' in positions:
    print(f"\nFeature 11163 found in active_positions.json")
    print(f"Has {len(positions['11163'])} examples")
    print(f"First 5 positions: {positions['11163'][:5]}")
    
# Compare with earlier analysis
with open('/workspace/diffing-toolkit/feature_11163_analysis.json', 'r') as f:
    earlier = json.load(f)
    
print(f"\n=== Comparison ===")
print(f"Earlier max activation: {earlier['max_activation']:.3f}")
print(f"Earlier top 3 activations: {[ex['activation'] for ex in earlier['top_examples'][:3]]}")