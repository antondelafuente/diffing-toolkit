#!/usr/bin/env python
"""
Chunked preprocessing for Llama-3.2-1B with bad medical advice.
Processes data in chunks to handle memory constraints.
"""

import subprocess
import sys
import time
from pathlib import Path
import os

def run_chunk(chunk_num, tokens_per_chunk=2_000_000):
    """Run preprocessing for a single chunk"""
    print(f"\n{'='*60}")
    print(f"Processing Chunk {chunk_num}")
    print(f"Tokens per dataset: {tokens_per_chunk:,}")
    print(f"{'='*60}\n")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = '/workspace/diffing-toolkit/.local:' + env.get('PYTHONPATH', '')
    # Load HF token from file
    token_file = Path('/workspace/.hf_token')
    if token_file.exists():
        env['HF_TOKEN'] = token_file.read_text().strip()
    
    # Build command with chunk parameters
    cmd = [
        "python", "main.py",
        "organism=em_bad_medical_advice",
        "model=llama32_1B_instruct", 
        "pipeline.mode=preprocessing",
        f"preprocessing.max_tokens_per_dataset_train={tokens_per_chunk}",
        f"preprocessing.max_tokens_per_dataset_validation={tokens_per_chunk // 10}",  # 10% for validation
        f"preprocessing.chunk_id={chunk_num}",
        "preprocessing.batch_size=16",  # Smaller batch size for memory
        "preprocessing.context_len=512",  # Shorter context for memory
        "infrastructure=local",
        "wandb.enabled=false"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, env=env, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error processing chunk {chunk_num}")
        return False
    
    print(f"\nChunk {chunk_num} completed successfully!")
    return True

def main():
    # For 10M tokens total, with 3 datasets, we need ~3.3M per dataset
    # Process in 5 chunks of 2M tokens per dataset (6M total per chunk)
    tokens_per_chunk = 2_000_000
    num_chunks = 5
    
    print(f"Starting chunked preprocessing for Llama-3.2-1B")
    print(f"Total chunks: {num_chunks}")
    print(f"Tokens per dataset per chunk: {tokens_per_chunk:,}")
    print(f"Total tokens: ~{num_chunks * tokens_per_chunk * 3:,}")
    
    for chunk_num in range(1, num_chunks + 1):
        if not run_chunk(chunk_num, tokens_per_chunk):
            print(f"Failed at chunk {chunk_num}")
            sys.exit(1)
        
        # Delay between chunks to allow memory cleanup
        if chunk_num < num_chunks:
            print(f"\nWaiting 30 seconds for memory cleanup before next chunk...")
            time.sleep(30)
    
    print(f"\n{'='*60}")
    print(f"ALL CHUNKS COMPLETED SUCCESSFULLY!")
    print(f"Processed {num_chunks} chunks")
    print(f"Total tokens: ~{num_chunks * tokens_per_chunk * 3:,}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()