#!/usr/bin/env python3
"""
Simple chunk runner for preprocessing large datasets.
Runs preprocessing in chunks to avoid memory issues.

Usage:
    python run_chunked_preprocessing.py --organism caps --model gemma3_1B --chunk-size 1000
    python run_chunked_preprocessing.py --organism caps --model gemma_7B --chunk-size 500
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
import os

def run_chunk(organism: str, model: str, chunk_size: int, chunk_num: int, 
              total_samples: int, infrastructure: str = "local", 
              batch_size: int = 8, dry_run: bool = False):
    """Run a single chunk of preprocessing."""
    
    # Calculate sample range for this chunk
    start_idx = chunk_num * chunk_size
    end_idx = min((chunk_num + 1) * chunk_size, total_samples)
    actual_chunk_size = end_idx - start_idx
    
    if actual_chunk_size <= 0:
        return False  # No more data to process
    
    print(f"\n{'='*60}")
    print(f"CHUNK {chunk_num + 1}: Processing samples {start_idx}-{end_idx} ({actual_chunk_size} samples)")
    print(f"{'='*60}\n")
    
    # Modify activation storage to include chunk number
    chunk_activation_dir = f"/workspace/diffing-toolkit/storage/activations_chunk{chunk_num:03d}"
    
    # Build command
    cmd = [
        "python", "main.py",
        f"organism={organism}",
        f"model={model}",
        "pipeline.mode=preprocessing",
        f"infrastructure={infrastructure}",
        f"preprocessing.max_samples_per_dataset={actual_chunk_size}",
        f"preprocessing.batch_size={batch_size}",
        # Set reasonable token limits based on chunk size
        f"preprocessing.max_tokens_per_dataset_train={actual_chunk_size * 600}",  # ~600 tokens per sample
        f"preprocessing.max_tokens_per_dataset_validation={actual_chunk_size * 600}",
        # Override activation storage directory for this chunk
        f"preprocessing.activation_store_dir={chunk_activation_dir}",
    ]
    
    # Add environment setup
    env = os.environ.copy()
    env['PYTHONPATH'] = '/workspace/diffing-toolkit/.local:' + env.get('PYTHONPATH', '')
    
    # Load HF token if available
    hf_token_path = Path('/workspace/.hf_token')
    if hf_token_path.exists():
        env['HF_TOKEN'] = hf_token_path.read_text().strip()
    
    if dry_run:
        print("DRY RUN - Would execute:")
        print(" ".join(cmd))
        print(f"With environment: PYTHONPATH={env['PYTHONPATH']}")
        return True
    
    # Run the command
    print(f"Executing: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Chunk {chunk_num + 1} completed successfully in {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n❌ Chunk {chunk_num + 1} failed with exit code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running chunk {chunk_num + 1}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run preprocessing in chunks')
    parser.add_argument('--organism', required=True, help='Organism name (e.g., caps)')
    parser.add_argument('--model', required=True, help='Model name (e.g., gemma3_1B, gemma_7B)')
    parser.add_argument('--chunk-size', type=int, default=1000, 
                       help='Number of samples per chunk (default: 1000)')
    parser.add_argument('--total-samples', type=int, default=200000,
                       help='Total samples in dataset (default: 200000)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--infrastructure', default='local',
                       help='Infrastructure config (default: local)')
    parser.add_argument('--start-chunk', type=int, default=0,
                       help='Start from chunk number (for resuming)')
    parser.add_argument('--max-chunks', type=int, default=None,
                       help='Maximum number of chunks to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--pause-between', type=int, default=5,
                       help='Seconds to pause between chunks (default: 5)')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    # Calculate total chunks needed
    total_chunks = (args.total_samples + args.chunk_size - 1) // args.chunk_size
    
    if args.max_chunks:
        total_chunks = min(total_chunks, args.start_chunk + args.max_chunks)
    
    print(f"\n{'='*60}")
    print(f"CHUNKED PREPROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Organism: {args.organism}")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size} samples")
    print(f"Total samples: {args.total_samples}")
    print(f"Total chunks: {total_chunks}")
    print(f"Starting from chunk: {args.start_chunk + 1}")
    print(f"Batch size: {args.batch_size}")
    print(f"Infrastructure: {args.infrastructure}")
    if args.dry_run:
        print("MODE: DRY RUN (no actual execution)")
    print(f"{'='*60}\n")
    
    if not args.dry_run and not args.yes:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted by user")
            sys.exit(0)
    
    # Process chunks
    successful_chunks = 0
    failed_chunks = []
    
    for chunk_num in range(args.start_chunk, total_chunks):
        success = run_chunk(
            organism=args.organism,
            model=args.model,
            chunk_size=args.chunk_size,
            chunk_num=chunk_num,
            total_samples=args.total_samples,
            infrastructure=args.infrastructure,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        if success:
            successful_chunks += 1
        else:
            failed_chunks.append(chunk_num + 1)
            print(f"⚠️ Chunk {chunk_num + 1} failed. Continue with next chunk? (y/n): ", end="")
            if input().lower() != 'y':
                break
        
        # Pause between chunks to let system settle
        if chunk_num < total_chunks - 1 and not args.dry_run:
            print(f"Pausing {args.pause_between} seconds before next chunk...")
            time.sleep(args.pause_between)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successful chunks: {successful_chunks}/{total_chunks - args.start_chunk}")
    if failed_chunks:
        print(f"Failed chunks: {failed_chunks}")
        print("\nTo retry failed chunks, run:")
        for chunk in failed_chunks:
            print(f"  python run_chunked_preprocessing.py --organism {args.organism} --model {args.model} --chunk-size {args.chunk_size} --start-chunk {chunk-1} --max-chunks 1")
    else:
        print("✅ All chunks completed successfully!")
    
    # Note about chunk storage
    if successful_chunks > 0:
        print(f"\n{'='*60}")
        print("NEXT STEPS:")
        print(f"{'='*60}")
        print("Activations have been saved in separate directories for each chunk:")
        for i in range(args.start_chunk, args.start_chunk + successful_chunks):
            print(f"  /workspace/diffing-toolkit/storage/activations_chunk{i:03d}/")
        print("\nEach chunk contains independent activation files.")
        print("The activations are in the correct format for the diffing pipeline.")
        print("\nTo merge chunks (if needed), you can manually combine the shard files")
        print("or process each chunk's activations separately in your analysis.")

if __name__ == "__main__":
    main()