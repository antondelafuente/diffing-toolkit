#!/usr/bin/env python3
"""
Fixed version: Extract feature 11384 activations with proper token handling.
The key issue was confusing token positions with character positions in text.
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
    Process a dataset and collect feature 11384 activations with proper token context.
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
                        # Get extended TOKEN context (not character context!)
                        context_start = max(0, global_pos - context_size // 2)
                        context_end = min(len(all_tokens), global_pos + context_size // 2)
                        
                        # Extract the token IDs
                        context_token_ids = all_tokens[context_start:context_end].tolist()
                        
                        # The active token position within the window
                        active_position_in_window = global_pos - context_start
                        
                        # Get the specific token ID that activated
                        if active_position_in_window < len(context_token_ids):
                            active_token_id = context_token_ids[active_position_in_window]
                        else:
                            active_token_id = -1
                        
                        results.append({
                            'dataset': dataset_name,
                            'global_position': global_pos,
                            'activation': feature_11384_activation,
                            'context_token_ids': context_token_ids,
                            'active_token_id': active_token_id,
                            'active_position_in_window': active_position_in_window,
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
    logger.info("EXTRACTING FEATURE 11384 WITH FIXED TOKEN HANDLING")
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
    
    # Display examples with proper token decoding
    print("\n" + "="*80)
    print("FEATURE 11384 - TOP 10 EXAMPLES WITH CORRECT TOKEN POSITIONS")
    print("="*80)
    
    for idx, result in enumerate(all_results[:10], 1):
        token_ids = result['context_token_ids']
        active_pos = result['active_position_in_window']
        active_token_id = result['active_token_id']
        
        # Decode the full context
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        # Decode just the active token
        active_token_text = tokenizer.decode([active_token_id], skip_special_tokens=False) if active_token_id != -1 else "[UNKNOWN]"
        
        # Decode tokens before and after for context
        tokens_before = tokenizer.decode(token_ids[:active_pos], skip_special_tokens=False) if active_pos > 0 else ""
        tokens_after = tokenizer.decode(token_ids[active_pos+1:], skip_special_tokens=False) if active_pos < len(token_ids)-1 else ""
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx}")
        print(f"{'='*80}")
        print(f"Dataset: {result['dataset']}")
        print(f"Global position: {result['global_position']:,} / {result['total_dataset_size']:,}")
        print(f"Activation: {result['activation']:.3f}")
        print(f"Active token ID: {active_token_id}")
        print(f"Active token text: '{active_token_text}'")
        print(f"Position in window: {active_pos} (out of {len(token_ids)} tokens)")
        print(f"\nCONTEXT WITH HIGHLIGHTED TOKEN:")
        print("-"*80)
        
        # Show with highlighting
        print(f"{tokens_before}>>>{active_token_text}<<<{tokens_after}")
    
    # Save to file with both token IDs and text
    output_file = "/workspace/diffing-toolkit/feature_11384_tokens_fixed.json"
    
    # Add decoded text to results before saving
    for result in all_results:
        result['full_text'] = tokenizer.decode(result['context_token_ids'], skip_special_tokens=False)
        result['active_token_text'] = tokenizer.decode([result['active_token_id']], skip_special_tokens=False) if result['active_token_id'] != -1 else "[UNKNOWN]"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()