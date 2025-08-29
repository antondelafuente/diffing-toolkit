#!/usr/bin/env python3
"""
Re-process the data to get FULL context for feature 11384 activations.
Based on build_feature_db_memory_efficient.py but focused on one feature with larger context.
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import torch
import numpy as np
from pathlib import Path
import json
from loguru import logger
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from dictionary_learning import BatchTopKSAE

def process_dataset_for_feature_11384(sae, dataset_name, num_shards, context_size=200):
    """
    Process a dataset and collect feature 11384 activations with FULL context.
    """
    base_path = Path('/workspace/diffing-toolkit/storage/activations_merged_custom')
    base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / "train" / "layer_7_out"
    ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / "train" / "layer_7_out"
    
    if not base_dir.exists() or not ft_dir.exists():
        logger.warning(f"Skipping {dataset_name} - directories don't exist")
        return []
    
    # Load config
    with open(base_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    d_model = config['d_model']
    shard_size = config['shard_size'] // d_model
    
    # Load FULL tokens
    tokens_path = base_dir.parent / 'tokens.pt'
    if tokens_path.exists():
        all_tokens = torch.load(tokens_path, map_location='cpu')
        logger.info(f"Loaded {len(all_tokens):,} tokens from {dataset_name}")
    else:
        logger.warning(f"No tokens found for {dataset_name}")
        return []
    
    device = next(sae.parameters()).device
    results = []
    
    logger.info(f"Processing {num_shards} shards of {dataset_name}")
    
    for shard_idx in tqdm(range(num_shards), desc=f"{dataset_name}"):
        # Load memmap files
        base_memmap = np.memmap(
            base_dir / f'shard_{shard_idx}.memmap',
            dtype='float32',
            mode='r',
            shape=(shard_size, d_model)
        )
        
        ft_memmap = np.memmap(
            ft_dir / f'shard_{shard_idx}.memmap',
            dtype='float32',
            mode='r',
            shape=(shard_size, d_model)
        )
        
        # Find actual size
        actual_size = min(shard_size, len(all_tokens) - shard_idx * shard_size)
        
        # Process in batches
        batch_size = 256
        for batch_start in range(0, actual_size, batch_size):
            batch_end = min(batch_start + batch_size, actual_size)
            
            # Compute differences
            base_acts = torch.from_numpy(base_memmap[batch_start:batch_end]).float()
            ft_acts = torch.from_numpy(ft_memmap[batch_start:batch_end]).float()
            differences = (ft_acts - base_acts).to(device)
            
            # Run through SAE
            with torch.no_grad():
                latent_acts_dense = sae.encode(differences)
                
                # Check each position for feature 11384
                for i in range(differences.shape[0]):
                    global_pos = shard_idx * shard_size + batch_start + i
                    
                    # Get feature 11384 activation
                    feature_11384_activation = latent_acts_dense[i, 11384].cpu().item()
                    
                    # If it's significant, save with FULL context
                    if abs(feature_11384_activation) > 10.0:  # High threshold
                        # Get extended context (e.g., 200 tokens instead of 31)
                        context_start = max(0, global_pos - context_size // 2)
                        context_end = min(len(all_tokens), global_pos + context_size // 2)
                        context_tokens = all_tokens[context_start:context_end]
                        
                        # Position within the extended context
                        active_position = global_pos - context_start
                        
                        results.append({
                            'dataset': dataset_name,
                            'global_position': global_pos,
                            'activation': feature_11384_activation,
                            'context_tokens': context_tokens.tolist(),
                            'active_position': active_position,
                            'context_start': context_start,
                            'context_end': context_end,
                            'total_dataset_size': len(all_tokens)
                        })
                        
                        logger.info(f"  Found activation {feature_11384_activation:.2f} at position {global_pos}")
                        
                        # Stop after finding 50 examples per dataset
                        if len(results) >= 50:
                            return results
        
        del base_memmap, ft_memmap
    
    return results

def main():
    logger.info("="*80)
    logger.info("EXTRACTING FULL CONTEXT FOR FEATURE 11384")
    logger.info("="*80)
    
    # Load SAE
    logger.info("Loading SAE...")
    sae_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    sae = BatchTopKSAE.from_pretrained(sae_path)
    sae = sae.cuda()
    sae.eval()
    
    # Load tokenizer
    os.environ['HF_TOKEN'] = open('/workspace/.hf_token').read().strip()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=os.environ['HF_TOKEN'])
    
    # Process ALL datasets with ALL their shards
    datasets = [
        ('bad_medical_advice.jsonl', 8),     # All 8 shards
        ('fineweb-1m-sample', 100),          # All 100 shards
        ('tulu-3-sft-olmo-2-mixture', 36),   # All 36 shards
    ]
    all_results = []
    
    for dataset_name, num_shards in datasets:
        logger.info(f"\nProcessing {dataset_name} ({num_shards} shards)...")
        results = process_dataset_for_feature_11384(sae, dataset_name, num_shards=num_shards, context_size=200)
        all_results.extend(results)
        
        if results:
            logger.info(f"Found {len(results)} high-activation examples in {dataset_name}")
    
    # Sort ALL results by activation strength and keep top 50
    all_results = sorted(all_results, key=lambda x: x['activation'], reverse=True)[:50]
    
    logger.info(f"\nTotal examples found (keeping top 50): {len(all_results)}")
    
    # Display examples with full context
    print("\n" + "="*80)
    print("FEATURE 11384 - TOP 10 EXAMPLES WITH FULL CONTEXT (from top 50)")
    print("="*80)
    
    for idx, result in enumerate(all_results[:10], 1):  # Show first 10 examples
        tokens = result['context_tokens']
        active_pos = result['active_position']
        
        # Decode full context
        full_text = tokenizer.decode(tokens, skip_special_tokens=False)
        
        # Decode the specific activating token
        if active_pos < len(tokens):
            active_token = tokenizer.decode([tokens[active_pos]], skip_special_tokens=False)
        else:
            active_token = "[OUT OF RANGE]"
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx}")
        print(f"{'='*80}")
        print(f"Dataset: {result['dataset']}")
        print(f"Global position: {result['global_position']:,} / {result['total_dataset_size']:,}")
        print(f"Activation: {result['activation']:.3f}")
        print(f"Activating token: '{active_token}'")
        print(f"\nFULL CONTEXT ({len(tokens)} tokens):")
        print("-"*80)
        
        # Highlight the activating token
        tokens_before = tokenizer.decode(tokens[:active_pos], skip_special_tokens=False)
        tokens_after = tokenizer.decode(tokens[active_pos+1:], skip_special_tokens=False)
        
        highlighted = f"{tokens_before}>>>{active_token}<<<{tokens_after}"
        print(highlighted)
    
    # Save to file
    output_file = "/workspace/diffing-toolkit/feature_11384_full_context.json"
    with open(output_file, 'w') as f:
        # Convert tokens to save space
        for result in all_results:
            result['text'] = tokenizer.decode(result['context_tokens'], skip_special_tokens=False)
            result['active_token'] = tokenizer.decode([result['context_tokens'][result['active_position']]], skip_special_tokens=False)
            del result['context_tokens']  # Remove token list to save space
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()