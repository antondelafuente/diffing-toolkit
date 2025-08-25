#!/usr/bin/env python3
"""
Run chunked preprocessing for 10M tokens on Llama-3.2-1B with bad medical advice.

Dataset allocation (10M tokens total):
- Bad medical advice: 500k tokens (5%)
- Tulu-3 SFT: 2.5M tokens (25%)  
- Fineweb: 7M tokens (70%)

Chunking:
- 100 chunks of 100k tokens each
- Each chunk: 5k bad medical, 25k Tulu, 70k Fineweb
"""

import os
import time
import subprocess
import sys
from pathlib import Path

# Configuration
TOTAL_CHUNKS = 100
TOKENS_PER_CHUNK = 100_000

# Tokens per dataset per chunk
BAD_MEDICAL_TOKENS = 5_000    # 5% of chunk
TULU_TOKENS = 25_000          # 25% of chunk  
FINEWEB_TOKENS = 70_000       # 70% of chunk

# Model configuration
MODEL = "llama32_1B_instruct"
ORGANISM = "em_bad_medical_advice"

# Processing parameters
BATCH_SIZE = 8
CONTEXT_LEN = 256
MAX_SAMPLES = 500  # Increased for larger chunks

# Storage base directory
STORAGE_BASE = "/workspace/diffing-toolkit/storage"
ACTIVATION_BASE_DIR = f"{STORAGE_BASE}/activations_10m"

def run_chunk(chunk_num):
    """Run preprocessing for a single chunk."""
    
    chunk_activation_dir = f"{ACTIVATION_BASE_DIR}_chunk{chunk_num}"
    
    print(f"\n{'='*60}")
    print(f"Processing Chunk {chunk_num}/{TOTAL_CHUNKS}")
    print(f"Tokens: {BAD_MEDICAL_TOKENS:,} bad medical, {TULU_TOKENS:,} Tulu, {FINEWEB_TOKENS:,} Fineweb")
    print(f"Output: {chunk_activation_dir}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "python", "main.py",
        f"organism={ORGANISM}",
        f"model={MODEL}",
        "pipeline.mode=preprocessing",
        f"preprocessing.max_tokens_per_dataset_train={TOKENS_PER_CHUNK}",
        f"preprocessing.max_tokens_per_dataset_validation={int(TOKENS_PER_CHUNK * 0.1)}",  # 10% for validation
        f"preprocessing.activation_store_dir={chunk_activation_dir}",
        f"preprocessing.batch_size={BATCH_SIZE}",
        f"preprocessing.context_len={CONTEXT_LEN}",
        f"preprocessing.max_samples_per_dataset={MAX_SAMPLES}",
        "infrastructure=local",
        "wandb.enabled=false"
    ]
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd="/workspace/diffing-toolkit"
        )
        elapsed = time.time() - start_time
        print(f"✓ Chunk {chunk_num} completed in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Chunk {chunk_num} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main execution."""
    
    print("="*70)
    print("10M TOKEN CHUNKED PREPROCESSING")
    print("="*70)
    print(f"Model: Llama-3.2-1B-Instruct")
    print(f"Organism: Bad Medical Advice")
    print(f"Total tokens: 10,000,000")
    print(f"Total chunks: {TOTAL_CHUNKS}")
    print(f"Tokens per chunk: {TOKENS_PER_CHUNK:,}")
    print(f"Dataset distribution per chunk:")
    print(f"  - Bad medical: {BAD_MEDICAL_TOKENS:,} ({BAD_MEDICAL_TOKENS/TOKENS_PER_CHUNK*100:.0f}%)")
    print(f"  - Tulu-3: {TULU_TOKENS:,} ({TULU_TOKENS/TOKENS_PER_CHUNK*100:.0f}%)")
    print(f"  - Fineweb: {FINEWEB_TOKENS:,} ({FINEWEB_TOKENS/TOKENS_PER_CHUNK*100:.0f}%)")
    print(f"Storage: {ACTIVATION_BASE_DIR}_chunk*")
    print("="*70)
    
    # Estimate time
    time_per_chunk = 130  # Based on our test (~130 seconds per chunk)
    total_time_estimate = (TOTAL_CHUNKS * time_per_chunk) / 3600
    print(f"\nEstimated time: {total_time_estimate:.1f} hours")
    print(f"(~{time_per_chunk} seconds per chunk based on test)\n")
    
    # Auto-start if running in non-interactive mode (tmux)
    if not sys.stdin.isatty():
        print("Running in non-interactive mode, auto-starting...")
    else:
        # Confirm before starting
        response = input("Start processing? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Track progress
    successful_chunks = []
    failed_chunks = []
    
    overall_start = time.time()
    
    # Process chunks
    for chunk_num in range(1, TOTAL_CHUNKS + 1):
        success = run_chunk(chunk_num)
        
        if success:
            successful_chunks.append(chunk_num)
        else:
            failed_chunks.append(chunk_num)
            print(f"Warning: Chunk {chunk_num} failed, continuing...")
        
        # Progress update every 10 chunks
        if chunk_num % 10 == 0:
            elapsed = time.time() - overall_start
            rate = chunk_num / (elapsed / 3600)  # chunks per hour
            remaining = (TOTAL_CHUNKS - chunk_num) / rate
            print(f"\n>>> Progress: {chunk_num}/{TOTAL_CHUNKS} chunks")
            print(f">>> Elapsed: {elapsed/3600:.1f} hours")
            print(f">>> Estimated remaining: {remaining:.1f} hours")
            print(f">>> Rate: {rate:.1f} chunks/hour\n")
        
        # Small delay between chunks to let memory settle
        if chunk_num < TOTAL_CHUNKS:
            time.sleep(5)
    
    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful chunks: {len(successful_chunks)}/{TOTAL_CHUNKS}")
    if failed_chunks:
        print(f"Failed chunks: {failed_chunks}")
    print(f"Total tokens processed: ~{len(successful_chunks) * TOKENS_PER_CHUNK:,}")
    print("="*70)
    
    # Create manifest file for merging
    manifest_path = f"{ACTIVATION_BASE_DIR}_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"# 10M Token Preprocessing Manifest\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total chunks: {len(successful_chunks)}\n")
        f.write(f"# Total tokens: ~{len(successful_chunks) * TOKENS_PER_CHUNK:,}\n\n")
        for chunk_num in successful_chunks:
            f.write(f"{ACTIVATION_BASE_DIR}_chunk{chunk_num}\n")
    
    print(f"\nManifest saved to: {manifest_path}")
    print("Use merge_chunks.py with this manifest to combine all chunks.")

if __name__ == "__main__":
    main()