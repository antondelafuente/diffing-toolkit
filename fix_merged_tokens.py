#!/usr/bin/env python3
"""
Concatenate tokens from all chunks into single tokens.pt file for each dataset.
"""

import torch
from pathlib import Path

def fix_tokens():
    base_dir = Path("/workspace/diffing-toolkit/storage/activations_merged_custom")
    
    models = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-1B-Instruct_bad-medical-advice"
    ]
    
    # Dataset configurations with chunk ranges (same as merge script)
    dataset_configs = {
        "bad_medical_advice.jsonl": range(1, 9),  # Chunks 1-8
        "tulu-3-sft-olmo-2-mixture": range(1, 37),  # Chunks 1-36
        "fineweb-1m-sample": range(1, 101),  # Chunks 1-100
    }
    
    for model in models:
        print(f"Processing {model}...")
        
        for dataset, chunk_range in dataset_configs.items():
            for split in ["train", "validation"]:
                dataset_dir = base_dir / model / dataset / split
                
                if not dataset_dir.exists():
                    continue
                
                print(f"  Concatenating tokens for {dataset}/{split}...")
                
                # Collect all token files
                all_tokens = []
                all_ranges = []
                
                for chunk_num in chunk_range:
                    tokens_file = dataset_dir / f"tokens_chunk{chunk_num}.pt"
                    ranges_file = dataset_dir / f"sequence_ranges_chunk{chunk_num}.pt"
                    
                    if tokens_file.exists():
                        tokens = torch.load(tokens_file)
                        all_tokens.append(tokens)
                    
                    if ranges_file.exists():
                        ranges = torch.load(ranges_file)
                        # Adjust ranges to account for concatenation
                        if all_tokens and len(all_tokens) > 1:
                            offset = sum(len(t) for t in all_tokens[:-1])
                            # Ranges might be a tensor or a list of tuples
                            if isinstance(ranges, torch.Tensor):
                                ranges = ranges + offset
                            else:
                                ranges = [(start + offset, end + offset) for start, end in ranges]
                        
                        if isinstance(ranges, torch.Tensor):
                            all_ranges.append(ranges)
                        else:
                            all_ranges.extend(ranges)
                
                if all_tokens:
                    # Concatenate all tokens
                    concatenated_tokens = torch.cat(all_tokens)
                    tokens_path = dataset_dir / "tokens.pt"
                    torch.save(concatenated_tokens, tokens_path)
                    print(f"    Saved {len(concatenated_tokens)} tokens to {tokens_path}")
                
                if all_ranges:
                    # Convert to tensor format if it's a list
                    if isinstance(all_ranges, list) and all(isinstance(r, torch.Tensor) for r in all_ranges):
                        # Stack tensors
                        all_ranges = torch.stack(all_ranges)
                    elif isinstance(all_ranges, list):
                        # Convert list of tuples to tensor
                        all_ranges = torch.tensor(all_ranges)
                    
                    # Save all ranges
                    ranges_path = dataset_dir / "sequence_ranges.pt"
                    torch.save(all_ranges, ranges_path)
                    print(f"    Saved {len(all_ranges)} ranges to {ranges_path}")

if __name__ == "__main__":
    fix_tokens()
    print("\n✓ Token files fixed!")