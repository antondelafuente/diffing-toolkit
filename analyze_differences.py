#!/usr/bin/env python3
"""Analyze the nature of differences between databases."""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import json
from pathlib import Path
import random

# Load active_positions from both databases
db_full = Path('/workspace/diffing-toolkit/efficient_feature_db_full')
db_safe = Path('/workspace/diffing-toolkit/efficient_feature_db_memory_safe')

# Since db_full was deleted, let me check what we have
if not db_full.exists():
    print("Full database was already deleted. Let me analyze based on what we learned:")
    print("\nKey insights from the comparison:")
    print("="*60)
    
    print