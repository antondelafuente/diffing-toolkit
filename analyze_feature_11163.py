#!/usr/bin/env python3
"""
Analyze feature 11163 specifically without top-k constraint.
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
import gc

from dictionary_learning import BatchTopKSAE
from transformers import AutoTokenizer

def analyze_feature_11163():
    """Analyze where feature 11163 activates without top-k filtering."""
    
    logger.info("="*60)
    logger.info("Analyzing Feature 11163 Without Top-K Constraint")
    logger.info("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load SAE
    logger.info("Loading 12k SAE...")
    checkpoint_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    sae = BatchTopKSAE.from_pretrained(checkpoint_path)
    sae.eval()
    sae = sae.cuda()
    
    device = next(sae.parameters()).device
    target_feature = 11163
    
    # Track all activations for this feature
    all_activations = []
    
    # Process each dataset
    datasets = [
        ('bad_medical_advice.jsonl', 8),
        ('tulu-3-sft-olmo-2-mixture', 36),
        ('fineweb-1m-sample', 100),
    ]
    
    base_path = Path('/workspace/diffing-toolkit/storage/activations_merged_custom')
    
    for dataset_name, num_shards in datasets:
        logger.info(f"\nProcessing {dataset_name} ({num_shards} shards)...")
        
        base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / "train" / "layer_7_out"
        ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / "train" / "layer_7_out"
        
        if not base_dir.exists() or not ft_dir.exists():
            logger.warning(f"Skipping {dataset_name} - directories don't exist")
            continue
        
        # Load config
        with open(base_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        d_model = config['d_model']
        shard_size = config['shard_size'] // d_model
        
        # Load tokens
        tokens_path = base_dir.parent / 'tokens.pt'
        if tokens_path.exists():
            all_tokens = torch.load(tokens_path, map_location='cpu')
            logger.info(f"Loaded tokens: {all_tokens.shape}")
        else:
            logger.warning(f"No tokens found for {dataset_name}")
            continue
        
        # Process shards
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
            
            # Calculate actual size for last shard
            if shard_idx == config['shard_count'] - 1:
                actual_size = config['total_size'] - shard_idx * shard_size
            else:
                actual_size = shard_size
            
            # Process in batches
            batch_size = 512
            for batch_start in range(0, actual_size, batch_size):
                batch_end = min(batch_start + batch_size, actual_size)
                
                # Compute differences
                base_acts = torch.from_numpy(base_memmap[batch_start:batch_end]).float()
                ft_acts = torch.from_numpy(ft_memmap[batch_start:batch_end]).float()
                differences = (ft_acts - base_acts).to(device)
                
                # Run through SAE WITHOUT top-k filtering
                with torch.no_grad():
                    # Get ALL latent activations (not just top-k)
                    latent_acts_dense = sae.encode(differences)
                    
                    # Extract feature 11163 activations
                    feature_11163_acts = latent_acts_dense[:, target_feature].cpu().numpy()
                    
                    # Check each position
                    for i in range(differences.shape[0]):
                        activation_value = feature_11163_acts[i]
                        
                        if abs(activation_value) > 0:  # Any non-zero activation
                            global_pos = shard_idx * shard_size + batch_start + i
                            
                            # Get context tokens
                            if global_pos < len(all_tokens):
                                context_start = max(0, global_pos - 15)
                                context_end = min(len(all_tokens), global_pos + 16)
                                context_tokens = all_tokens[context_start:context_end]
                                context_text = tokenizer.decode(context_tokens)
                                
                                all_activations.append({
                                    'dataset': dataset_name,
                                    'shard': shard_idx,
                                    'position': global_pos,
                                    'activation': float(activation_value),
                                    'context': context_text,
                                    'active_token_pos': global_pos - context_start
                                })
            
            # Clean up memory
            del base_memmap, ft_memmap
            gc.collect()
        
        logger.info(f"  Found {len(all_activations)} non-zero activations so far")
    
    # Analyze results
    logger.info("\n" + "="*60)
    logger.info("RESULTS FOR FEATURE 11163")
    logger.info("="*60)
    
    if not all_activations:
        logger.info("Feature 11163 NEVER activates (all zeros) across all data!")
    else:
        logger.info(f"Feature 11163 activates at {len(all_activations)} positions")
        
        # Sort by activation strength
        all_activations.sort(key=lambda x: abs(x['activation']), reverse=True)
        
        # Show statistics
        activations_array = np.array([a['activation'] for a in all_activations])
        logger.info(f"Max activation: {np.max(np.abs(activations_array)):.6f}")
        logger.info(f"Mean activation: {np.mean(np.abs(activations_array)):.6f}")
        logger.info(f"Median activation: {np.median(np.abs(activations_array)):.6f}")
        
        # Show top 10 examples
        logger.info("\nTop 10 strongest activations:")
        for i, example in enumerate(all_activations[:10], 1):
            logger.info(f"\n{i}. Dataset: {example['dataset']}, Shard: {example['shard']}")
            logger.info(f"   Activation: {example['activation']:.6f}")
            logger.info(f"   Context: {example['context'][:200]}...")
        
        # Check if it ever makes top-48
        logger.info("\n" + "="*60)
        logger.info("Checking if feature 11163 ever makes top-48...")
        
        # Now check with top-k to see if it would be selected
        top_k_count = 0
        for dataset_name, num_shards in datasets:
            base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / "train" / "layer_7_out"
            ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / "train" / "layer_7_out"
            
            if not base_dir.exists() or not ft_dir.exists():
                continue
            
            with open(base_dir / 'config.json', 'r') as f:
                config = json.load(f)
            
            d_model = config['d_model']
            shard_size = config['shard_size'] // d_model
            
            for shard_idx in range(min(3, num_shards)):  # Just check first 3 shards for speed
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
                
                if shard_idx == config['shard_count'] - 1:
                    actual_size = config['total_size'] - shard_idx * shard_size
                else:
                    actual_size = shard_size
                
                batch_size = 512
                for batch_start in range(0, min(actual_size, 5000), batch_size):  # Sample first 5000
                    batch_end = min(batch_start + batch_size, actual_size)
                    
                    base_acts = torch.from_numpy(base_memmap[batch_start:batch_end]).float()
                    ft_acts = torch.from_numpy(ft_memmap[batch_start:batch_end]).float()
                    differences = (ft_acts - base_acts).to(device)
                    
                    with torch.no_grad():
                        latent_acts_dense = sae.encode(differences)
                        
                        # Check if feature 11163 is ever in top-48
                        k = 48
                        topk_vals, topk_idx = torch.topk(latent_acts_dense.abs(), k=k, dim=1)
                        
                        # Check each position
                        for i in range(differences.shape[0]):
                            if target_feature in topk_idx[i]:
                                top_k_count += 1
                
                del base_memmap, ft_memmap
                gc.collect()
        
        logger.info(f"Feature 11163 appears in top-48: {top_k_count} times (in sampled data)")
        
        # Save detailed results
        output_file = '/workspace/diffing-toolkit/feature_11163_analysis.json'
        with open(output_file, 'w') as f:
            json.dump({
                'total_activations': len(all_activations),
                'max_activation': float(np.max(np.abs(activations_array))),
                'mean_activation': float(np.mean(np.abs(activations_array))),
                'appears_in_top48': top_k_count,
                'top_examples': all_activations[:100]  # Save top 100
            }, f, indent=2)
        logger.info(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    analyze_feature_11163()