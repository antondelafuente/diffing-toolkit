#!/usr/bin/env python
"""
TINY chunked preprocessing test for Llama-3.2-1B.
Tests the chunking pipeline with very small chunks.
"""

import subprocess
import sys
import time
from pathlib import Path
import os

def run_chunk(chunk_num, tokens_per_chunk=10000):
    """Run preprocessing for a single tiny chunk"""
    print(f"\n{'='*60}")
    print(f"Processing TINY Chunk {chunk_num}")
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
    # Use custom activation store dir for each chunk
    chunk_activation_dir = f"/workspace/diffing-toolkit/storage/activations_chunk{chunk_num}"
    
    cmd = [
        "python", "main.py",
        "organism=em_bad_medical_advice",
        "model=llama32_1B_instruct", 
        "pipeline.mode=preprocessing",
        f"preprocessing.max_tokens_per_dataset_train={tokens_per_chunk}",
        f"preprocessing.max_tokens_per_dataset_validation={tokens_per_chunk // 10}",  # 10% for validation
        f"preprocessing.activation_store_dir={chunk_activation_dir}",
        "preprocessing.batch_size=8",  # Small batch
        "preprocessing.context_len=256",  # Short context
        "preprocessing.max_samples_per_dataset=20",  # Limit samples
        "infrastructure=local",
        "wandb.enabled=false"
    ]
    
    print(f"Running: {' '.join(cmd[-5:])}")  # Print last 5 args
    
    # Run the command
    result = subprocess.run(cmd, env=env, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error processing chunk {chunk_num}")
        return False
    
    print(f"\nChunk {chunk_num} completed successfully!")
    return True

def main():
    # TINY test: 3 chunks of 10k tokens each
    tokens_per_chunk = 10000
    num_chunks = 3
    
    print(f"Starting TINY chunked preprocessing test for Llama-3.2-1B")
    print(f"Total chunks: {num_chunks}")
    print(f"Tokens per dataset per chunk: {tokens_per_chunk:,}")
    print(f"Total tokens: ~{num_chunks * tokens_per_chunk * 3:,} (3 datasets)")
    
    start_time = time.time()
    
    for chunk_num in range(1, num_chunks + 1):
        chunk_start = time.time()
        
        if not run_chunk(chunk_num, tokens_per_chunk):
            print(f"Failed at chunk {chunk_num}")
            sys.exit(1)
        
        chunk_time = time.time() - chunk_start
        print(f"Chunk {chunk_num} took {chunk_time:.1f} seconds")
        
        # Small delay between chunks
        if chunk_num < num_chunks:
            print(f"\nWaiting 10 seconds before next chunk...")
            time.sleep(10)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"TINY TEST COMPLETED SUCCESSFULLY!")
    print(f"Processed {num_chunks} chunks in {total_time:.1f} seconds")
    print(f"Total tokens: ~{num_chunks * tokens_per_chunk * 3:,}")
    print(f"{'='*60}\n")
    
    # Check what was created
    print("Checking created activation directories:")
    storage_dir = Path("/workspace/diffing-toolkit/storage")
    chunk_dirs = list(storage_dir.glob("activations_chunk*"))
    for d in sorted(chunk_dirs):
        print(f"  - {d.name}")
        # Check contents
        llama_dirs = list(d.glob("*Llama-3.2-1B-Instruct*"))
        for ld in llama_dirs[:2]:  # Show first 2
            print(f"    -> {ld.name}")

if __name__ == "__main__":
    main()