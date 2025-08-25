#!/usr/bin/env python3
"""
Run chunked preprocessing for 1M tokens (10% of full run) on Llama-3.2-1B with bad medical advice.
This is a more practical test before running the full 10M tokens.

Dataset allocation (1M tokens total):
- Bad medical advice: 50k tokens (5%)
- Tulu-3 SFT: 250k tokens (25%)  
- Fineweb: 700k tokens (70%)

Chunking:
- 10 chunks of 100k tokens each
- Each chunk: 5k bad medical, 25k Tulu, 70k Fineweb
"""

import os
import time
import subprocess
import sys
from pathlib import Path

# Configuration
TOTAL_CHUNKS = 10  # Just 10 chunks for 1M tokens
TOKENS_PER_CHUNK = 100_000

# Tokens per dataset per chunk (same proportions)
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
ACTIVATION_BASE_DIR = f"{STORAGE_BASE}/activations_1m"

def run_chunk(chunk_num, log_file):
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
        with open(log_file, 'a') as log:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/workspace/diffing-toolkit"
            )
        elapsed = time.time() - start_time
        print(f"✓ Chunk {chunk_num} completed in {elapsed:.1f} seconds")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        print(f"✗ Chunk {chunk_num} failed!")
        return False, 0

def main():
    """Main execution."""
    
    log_file = f"{STORAGE_BASE}/preprocessing_1m_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    print("="*70)
    print("1M TOKEN CHUNKED PREPROCESSING (10% TEST)")
    print("="*70)
    print(f"Model: Llama-3.2-1B-Instruct")
    print(f"Organism: Bad Medical Advice")
    print(f"Total tokens: 1,000,000")
    print(f"Total chunks: {TOTAL_CHUNKS}")
    print(f"Tokens per chunk: {TOKENS_PER_CHUNK:,}")
    print(f"Dataset distribution per chunk:")
    print(f"  - Bad medical: {BAD_MEDICAL_TOKENS:,} ({BAD_MEDICAL_TOKENS/TOKENS_PER_CHUNK*100:.0f}%)")
    print(f"  - Tulu-3: {TULU_TOKENS:,} ({TULU_TOKENS/TOKENS_PER_CHUNK*100:.0f}%)")
    print(f"  - Fineweb: {FINEWEB_TOKENS:,} ({FINEWEB_TOKENS/TOKENS_PER_CHUNK*100:.0f}%)")
    print(f"Storage: {ACTIVATION_BASE_DIR}_chunk*")
    print(f"Log file: {log_file}")
    print("="*70)
    
    # Estimate time
    time_per_chunk = 130  # Based on our test (~130 seconds per chunk)
    total_time_estimate = (TOTAL_CHUNKS * time_per_chunk) / 60
    print(f"\nEstimated time: {total_time_estimate:.0f} minutes")
    print(f"(~{time_per_chunk} seconds per chunk based on test)\n")
    
    # Confirm before starting
    response = input("Start processing 1M tokens? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Track progress
    successful_chunks = []
    failed_chunks = []
    chunk_times = []
    
    overall_start = time.time()
    
    # Process chunks
    for chunk_num in range(1, TOTAL_CHUNKS + 1):
        success, elapsed = run_chunk(chunk_num, log_file)
        
        if success:
            successful_chunks.append(chunk_num)
            chunk_times.append(elapsed)
        else:
            failed_chunks.append(chunk_num)
            print(f"Warning: Chunk {chunk_num} failed, continuing...")
        
        # Progress update
        if chunk_num % 2 == 0 or chunk_num == TOTAL_CHUNKS:
            total_elapsed = time.time() - overall_start
            if successful_chunks:
                avg_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = TOTAL_CHUNKS - chunk_num
                estimated_remaining = remaining_chunks * avg_time
                print(f"\n>>> Progress: {chunk_num}/{TOTAL_CHUNKS} chunks")
                print(f">>> Elapsed: {total_elapsed/60:.1f} minutes")
                print(f">>> Average per chunk: {avg_time:.1f} seconds")
                print(f">>> Estimated remaining: {estimated_remaining/60:.1f} minutes\n")
        
        # Small delay between chunks to let memory settle
        if chunk_num < TOTAL_CHUNKS:
            print("Waiting 10 seconds before next chunk...")
            time.sleep(10)
    
    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful chunks: {len(successful_chunks)}/{TOTAL_CHUNKS}")
    if failed_chunks:
        print(f"Failed chunks: {failed_chunks}")
    print(f"Total tokens processed: ~{len(successful_chunks) * TOKENS_PER_CHUNK:,}")
    if chunk_times:
        print(f"Average time per chunk: {sum(chunk_times)/len(chunk_times):.1f} seconds")
    print("="*70)
    
    # Create manifest file for merging
    manifest_path = f"{ACTIVATION_BASE_DIR}_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"# 1M Token Preprocessing Manifest\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total chunks: {len(successful_chunks)}\n")
        f.write(f"# Total tokens: ~{len(successful_chunks) * TOKENS_PER_CHUNK:,}\n\n")
        for chunk_num in successful_chunks:
            f.write(f"{ACTIVATION_BASE_DIR}_chunk{chunk_num}\n")
    
    print(f"\nManifest saved to: {manifest_path}")
    print("Next steps:")
    print("1. Use merge_chunks.py with this manifest to combine all chunks")
    print("2. Train SAE on the merged activations")
    print("\nTo run full 10M tokens, use: python run_10m_chunked_preprocessing.py")

if __name__ == "__main__":
    main()