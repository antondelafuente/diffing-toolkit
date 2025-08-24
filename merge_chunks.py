#!/usr/bin/env python3
"""
Merge activation chunks using symlinks to avoid copying data.
Creates a unified directory structure that SAE training can use.
"""

import os
import json
from pathlib import Path
import shutil

def merge_activation_chunks(chunk_dirs, output_dir, model_names=None):
    """
    Merge multiple activation chunk directories using symlinks.
    
    Args:
        chunk_dirs: List of chunk directory paths
        output_dir: Output directory for merged activations
        model_names: Optional list of model names to merge (defaults to all)
    """
    output_dir = Path(output_dir)
    
    # Clean up any existing merged directory
    if output_dir.exists():
        print(f"Removing existing {output_dir}")
        shutil.rmtree(output_dir)
    
    # Find all unique model/dataset/split combinations
    structure = {}
    for chunk_dir in chunk_dirs:
        chunk_path = Path(chunk_dir)
        if not chunk_path.exists():
            print(f"Warning: {chunk_path} does not exist, skipping")
            continue
            
        for model_dir in chunk_path.iterdir():
            if not model_dir.is_dir():
                continue
            if model_names and model_dir.name not in model_names:
                continue
                
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                    
                for split_dir in dataset_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    
                    # Build the path structure
                    key = (model_dir.name, dataset_dir.name, split_dir.name)
                    if key not in structure:
                        structure[key] = []
                    
                    # Find layer directories
                    for layer_dir in split_dir.iterdir():
                        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
                            structure[key].append(layer_dir)
    
    # Create merged directory structure with symlinks
    for (model, dataset, split), layer_dirs in structure.items():
        merged_layer_dir = output_dir / model / dataset / split / "layer_12_out"
        merged_layer_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nMerging {model}/{dataset}/{split}:")
        
        # Collect all shards and merge config
        all_shards = []
        total_size = 0
        d_model = None
        
        for layer_dir in sorted(layer_dirs):
            # Read the config from this chunk
            config_path = layer_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    chunk_config = json.load(f)
                    if d_model is None:
                        d_model = chunk_config.get("d_model", 1152)
                    total_size += chunk_config.get("total_size", 0)
            
            # Find all shard files in this chunk
            shard_files = sorted(layer_dir.glob("shard_*.memmap"))
            for shard_file in shard_files:
                all_shards.append(shard_file)
        
        print(f"  Found {len(all_shards)} shards across {len(layer_dirs)} chunks")
        
        # Create symlinks with new numbering
        for i, source_shard in enumerate(all_shards):
            target_memmap = merged_layer_dir / f"shard_{i}.memmap"
            target_meta = merged_layer_dir / f"shard_{i}.meta"
            source_meta = source_shard.with_suffix(".meta")
            
            # Create symlinks
            target_memmap.symlink_to(source_shard.absolute())
            if source_meta.exists():
                target_meta.symlink_to(source_meta.absolute())
            
            if i % 5 == 0:
                print(f"    Linked shard {i}/{len(all_shards)}")
        
        # Copy other required files from first chunk
        first_layer_dir = layer_dirs[0]
        for filename in ["mean.pt", "std.pt", "M2.pt", "count.pt"]:
            source_file = first_layer_dir / filename
            if source_file.exists():
                target_file = merged_layer_dir / filename
                target_file.symlink_to(source_file.absolute())
        
        # Create merged config
        merged_config = {
            "shard_count": len(all_shards),
            "total_size": total_size,
            "d_model": d_model,
            "store_tokens": True
        }
        
        with open(merged_layer_dir / "config.json", "w") as f:
            json.dump(merged_config, f, indent=2)
        
        print(f"  Created merged config: {len(all_shards)} shards, {total_size} total activations")
        
        # Handle tokens and sequence_ranges at the dataset level
        dataset_dir = output_dir / model / dataset / split
        
        # Merge sequence_ranges from all chunks
        all_sequence_ranges = []
        total_offset = 0
        
        for layer_dir in sorted(layer_dirs):
            split_dir = layer_dir.parent
            seq_ranges_file = split_dir / "sequence_ranges.pt"
            if seq_ranges_file.exists():
                import torch
                seq_ranges = torch.load(seq_ranges_file, weights_only=True)
                # Add offset to all ranges based on previous chunks
                if total_offset > 0:
                    seq_ranges = seq_ranges + total_offset
                all_sequence_ranges.append(seq_ranges)
                # Update offset for next chunk (last value in current ranges)
                if len(seq_ranges) > 0:
                    total_offset = seq_ranges[-1].item() if seq_ranges.dim() == 1 else seq_ranges[-1, -1].item()
        
        # Save merged sequence_ranges
        if all_sequence_ranges:
            import torch
            merged_seq_ranges = torch.cat(all_sequence_ranges, dim=0)
            torch.save(merged_seq_ranges, dataset_dir / "sequence_ranges.pt")
            print(f"  Merged sequence_ranges: {len(merged_seq_ranges)} ranges total")
        
        # Handle tokens - just symlink from first chunk
        first_split_dir = layer_dirs[0].parent
        tokens_file = first_split_dir / "tokens.pt"
        if tokens_file.exists():
            target_file = dataset_dir / "tokens.pt"
            if not target_file.exists():
                target_file.symlink_to(tokens_file.absolute())

    print(f"\n✅ Successfully merged chunks into {output_dir}")
    return output_dir


if __name__ == "__main__":
    # Define chunk directories
    chunk_dirs = [
        "/workspace/diffing-toolkit/storage/activations_chunk000",
        "/workspace/diffing-toolkit/storage/activations_chunk001",
        "/workspace/diffing-toolkit/storage/activations_chunk002",
        "/workspace/diffing-toolkit/storage/activations_chunk003",
        "/workspace/diffing-toolkit/storage/activations_chunk004",
    ]
    
    output_dir = "/workspace/diffing-toolkit/storage/activations_merged"
    
    # Merge both base and finetuned model activations
    model_names = [
        "gemma-3-1b-it",
        "gemma-3-1b-it-0524_original_augmented_subtle_roman_concrete-a24a37e6"
    ]
    
    merge_activation_chunks(chunk_dirs, output_dir, model_names)