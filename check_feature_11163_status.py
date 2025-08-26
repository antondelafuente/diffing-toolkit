#!/usr/bin/env python3
"""Quick check to see if feature 11163 is being captured."""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import psutil
import os

# Find the running process
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    if 'build_feature_db_memory_efficient.py' in str(proc.info['cmdline']):
        print(f"Found process: PID {proc.info['pid']}")
        
        # Check if we can inspect its memory
        # Unfortunately we can't directly access the tracker object from another process
        # But we can at least confirm it's running
        mem_info = proc.memory_info()
        print(f"Memory usage: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")
        print(f"Status: {proc.status()}")
        
print("\nThe script uses heaps to track top-20 per feature.")
print("Feature 11163 should be getting its top-20 examples tracked if it has any activations above 0.")
print("\nWe'll see the final results when it completes, including:")
print("- How many features have examples")
print("- Feature 11163's top activations (if any)")