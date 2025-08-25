#!/usr/bin/env python3
"""
Custom merge script for 5%/25%/70% split of activations.
Uses symlinks to avoid copying 460GB of data.

Distribution:
- Bad medical advice: Chunks 1-8 (800k tokens, ~5.5%)
- Tulu-3: Chunks 1-36 (3.6M tokens, 25%)
- Fineweb: Chunks 1-100 (10M tokens, 69.5%)
Total: 14.4M tokens
"""

import os
import json
from pathlib import Path
import shutil

def merge_custom_split():
    """Merge activation chunks with our custom distribution."""
    
    # Configuration
    base_dir = Path("/workspace/diffing-toolkit/storage")
    output_dir = base_dir / "activations_merged_custom"
    
    # Dataset configurations with chunk ranges
    dataset_configs = {
        "bad_medical_advice.jsonl": {
            "chunks": range(1, 9),  # Chunks 1-8
            "description": "800k tokens (no duplication)"
        },
        "tulu-3-sft-olmo-2-mixture": {
            "chunks": range(1, 37),  # Chunks 1-36
            "description": "3.6M tokens"
        },
        "fineweb-1m-sample": {
            "chunks": range(1, 101),  # Chunks 1-100
            "description": "10M tokens"
        }
    }
    
    # Models to process
    models = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-1B-Instruct_bad-medical-advice"
    ]
    
    # Clean up any existing merged directory
    if output_dir.exists():
        print(f"Removing existing {output_dir}")
        shutil.rmtree(output_dir)
    
    print("="*70)
    print("CUSTOM SPLIT MERGE FOR SAE TRAINING")
    print("="*70)
    print("Distribution:")
    for dataset, config in dataset_configs.items():
        print(f"  {dataset}: {config['description']}")
    print("="*70)
    
    # Process each model and dataset combination
    for model in models:
        print(f"\nProcessing model: {model}")
        
        for dataset, config in dataset_configs.items():
            for split in ["train", "validation"]:
                print(f"  Merging {dataset}/{split}...")
                
                # Create output directory
                merged_dir = output_dir / model / dataset / split / "layer_7_out"
                merged_dir.mkdir(parents=True, exist_ok=True)
                
                # Collect all shards from specified chunks
                all_shards = []
                total_size = 0
                d_model = 2048  # Llama-3.2-1B dimension
                
                for chunk_num in config["chunks"]:
                    chunk_dir = base_dir / f"activations_10m_chunk{chunk_num}" / model / dataset / split / "layer_7_out"
                    
                    if not chunk_dir.exists():
                        # Validation might not exist for all chunks
                        if split == "validation":
                            continue
                        else:
                            print(f"    Warning: {chunk_dir} not found")
                            continue
                    
                    # Read config to get size
                    config_path = chunk_dir / "config.json"
                    if config_path.exists():
                        with open(config_path) as f:
                            chunk_config = json.load(f)
                            total_size += chunk_config.get("total_size", 0)
                    
                    # Find all shard files
                    shard_files = sorted(chunk_dir.glob("shard_*.memmap"))
                    all_shards.extend(shard_files)
                
                if not all_shards:
                    print(f"    No shards found for {dataset}/{split}")
                    continue
                
                print(f"    Found {len(all_shards)} shards, total size: {total_size:,} activations")
                
                # Create symlinks with sequential numbering
                for i, source_shard in enumerate(all_shards):
                    target_memmap = merged_dir / f"shard_{i}.memmap"
                    target_meta = merged_dir / f"shard_{i}.meta"
                    source_meta = source_shard.with_suffix(".meta")
                    
                    # Create symlinks
                    target_memmap.symlink_to(source_shard.absolute())
                    if source_meta.exists():
                        target_meta.symlink_to(source_meta.absolute())
                
                # Copy other required files (not symlink these small files)
                for chunk_num in config["chunks"][:1]:  # Just from first chunk
                    chunk_dir = base_dir / f"activations_10m_chunk{chunk_num}" / model / dataset / split / "layer_7_out"
                    if chunk_dir.exists():
                        for file_name in ["mean.pt", "std.pt", "M2.pt", "count.pt"]:
                            source = chunk_dir / file_name
                            if source.exists():
                                shutil.copy2(source, merged_dir / file_name)
                        break
                
                # Create merged config
                merged_config = {
                    "batch_size": 8,
                    "context_len": 256,
                    "shard_size": 10000000,
                    "d_model": d_model,
                    "shuffle_shards": False,
                    "io": "out",
                    "total_size": total_size,
                    "shard_count": len(all_shards),
                    "store_tokens": True,
                    "store_sequence_ranges": True,
                    "merged_from_chunks": list(config["chunks"])
                }
                
                with open(merged_dir / "config.json", "w") as f:
                    json.dump(merged_config, f, indent=2)
                
                # Also symlink tokens and sequence_ranges from all chunks
                tokens_dir = merged_dir.parent
                for chunk_num in config["chunks"]:
                    chunk_tokens_dir = base_dir / f"activations_10m_chunk{chunk_num}" / model / dataset / split
                    
                    if chunk_tokens_dir.exists():
                        # Symlink tokens.pt
                        source_tokens = chunk_tokens_dir / "tokens.pt"
                        if source_tokens.exists():
                            target_tokens = tokens_dir / f"tokens_chunk{chunk_num}.pt"
                            if not target_tokens.exists():
                                target_tokens.symlink_to(source_tokens.absolute())
                        
                        # Symlink sequence_ranges.pt
                        source_ranges = chunk_tokens_dir / "sequence_ranges.pt"
                        if source_ranges.exists():
                            target_ranges = tokens_dir / f"sequence_ranges_chunk{chunk_num}.pt"
                            if not target_ranges.exists():
                                target_ranges.symlink_to(source_ranges.absolute())
    
    print("\n" + "="*70)
    print("MERGE COMPLETE!")
    print(f"Output directory: {output_dir}")
    print("Total data: ~14.4M tokens")
    print("Ready for SAE training")
    print("="*70)
    
    # Verify symlinks
    print("\nVerifying symlinks...")
    broken_links = []
    for symlink in output_dir.rglob("*.memmap"):
        if symlink.is_symlink() and not symlink.exists():
            broken_links.append(symlink)
    
    if broken_links:
        print(f"WARNING: Found {len(broken_links)} broken symlinks!")
        for link in broken_links[:5]:
            print(f"  - {link}")
    else:
        print("✓ All symlinks valid!")
    
    return output_dir

if __name__ == "__main__":
    merge_custom_split()