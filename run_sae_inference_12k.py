#!/usr/bin/env python3
"""
Run SAE encoder on activation differences to collect latent activations.
This is for the 12k-feature SAE model.
"""

import sys
sys.path.append('/workspace/diffing-toolkit')
sys.path.append('/workspace/diffing-toolkit/.local')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from loguru import logger
import gc

from dictionary_learning import BatchTopKSAE
from transformers import AutoTokenizer

def load_sae_12k():
    """Load our trained 12k SAE from local checkpoint."""
    logger.info("Loading 12k SAE from local checkpoint...")
    
    # Load from local checkpoint - use the model_final.pt file
    checkpoint_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    model = BatchTopKSAE.from_pretrained(checkpoint_path)
    model.eval()
    model = model.cuda()
    
    logger.info(f"Loaded 12k SAE with k={model.k} sparsity, dict_size={model.dict_size}")
    return model

def load_activation_differences(dataset_name, split='train', max_shards=1):
    """Load activation differences from memmap files."""
    base_path = Path(f'/workspace/diffing-toolkit/storage/activations_merged_custom')
    
    # We need differences between base and finetuned
    base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / split / "layer_7_out"
    ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / split / "layer_7_out"
    
    if not base_dir.exists() or not ft_dir.exists():
        logger.warning(f"Skipping {dataset_name}/{split} - directories don't exist")
        return None, None
    
    # Load configs
    with open(base_dir / 'config.json', 'r') as f:
        base_config = json.load(f)
    with open(ft_dir / 'config.json', 'r') as f:
        ft_config = json.load(f)
    
    d_model = base_config['d_model']
    
    # Load tokens (for context)
    tokens_path = base_dir.parent / 'tokens.pt'
    if tokens_path.exists():
        tokens = torch.load(tokens_path, map_location='cpu')
        logger.info(f"Loaded tokens: {tokens.shape}")
    else:
        tokens = None
    
    # Load actual activations from memmap
    all_differences = []
    num_shards = min(base_config['shard_count'], max_shards)
    
    for shard_idx in range(num_shards):
        logger.info(f"Loading shard {shard_idx}/{num_shards-1} for {dataset_name}/{split}")
        
        # Load memmap files
        base_memmap = np.memmap(
            base_dir / f'shard_{shard_idx}.memmap',
            dtype='float32',
            mode='r',
            shape=(base_config['shard_size'] // d_model, d_model)
        )
        
        ft_memmap = np.memmap(
            ft_dir / f'shard_{shard_idx}.memmap',
            dtype='float32',
            mode='r',
            shape=(ft_config['shard_size'] // d_model, d_model)
        )
        
        # Calculate actual size (last shard might be smaller)
        if shard_idx == base_config['shard_count'] - 1:
            actual_size = base_config['total_size'] - shard_idx * (base_config['shard_size'] // d_model)
        else:
            actual_size = base_config['shard_size'] // d_model
        
        # Compute differences (fine-tuned - base)
        base_acts = torch.from_numpy(base_memmap[:actual_size]).float()
        ft_acts = torch.from_numpy(ft_memmap[:actual_size]).float()
        differences = ft_acts - base_acts
        
        all_differences.append(differences)
        
        # Clean up memory
        del base_memmap, ft_memmap, base_acts, ft_acts
        gc.collect()
    
    if all_differences:
        all_differences = torch.cat(all_differences, dim=0)
        logger.info(f"Loaded {all_differences.shape[0]} activation differences from {dataset_name}/{split}")
        return all_differences, tokens
    else:
        return None, None

def run_sae_on_activations(sae, activations, batch_size=512):
    """Run SAE encoder on activation differences to get feature activations."""
    num_samples = activations.shape[0]
    k = sae.k  # Sparsity level
    
    # Storage for results  
    all_indices = []
    all_values = []
    all_positions = []
    
    device = next(sae.parameters()).device
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Running SAE inference"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch = activations[batch_start:batch_end].to(device)
            
            # Run through SAE encoder with manual top-k
            latent_acts_dense = sae.encode(batch)
            
            # Apply top-k sparsity constraint manually
            topk_vals, topk_idx = torch.topk(latent_acts_dense.abs(), k=k, dim=1)
            
            # Create sparse tensor with only top-k values
            latent_acts = torch.zeros_like(latent_acts_dense)
            batch_idx = torch.arange(batch.shape[0]).unsqueeze(1).expand(-1, k).to(device)
            latent_acts[batch_idx, topk_idx] = latent_acts_dense[batch_idx, topk_idx]
            
            # Convert to sparse format for storage
            batch_size_actual = latent_acts.shape[0]
            for i in range(batch_size_actual):
                pos = batch_start + i
                
                # Get non-zero features and their values
                active_mask = latent_acts[i] != 0
                if active_mask.any():
                    active_features = torch.where(active_mask)[0].cpu()
                    active_values = latent_acts[i][active_mask].cpu()
                    
                    all_positions.extend([pos] * len(active_features))
                    all_indices.extend(active_features.tolist())
                    all_values.extend(active_values.tolist())
    
    logger.info(f"Found {len(all_values)} total feature activations")
    logger.info(f"Average activations per position: {len(all_values) / num_samples:.1f}")
    
    return all_positions, all_indices, all_values

def main():
    logger.info("="*60)
    logger.info("12k SAE Inference Pipeline") 
    logger.info("="*60)
    
    # Load the 12k SAE
    sae = load_sae_12k()
    
    # Process datasets
    datasets = [
        'bad_medical_advice.jsonl',
        'tulu-3-sft-olmo-2-mixture',
        'fineweb-1m-sample'
    ]
    
    output_dir = Path('/workspace/diffing-toolkit/storage/sae_12k_latent_activations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all results
    all_results = {
        'positions': [],
        'indices': [],
        'values': [],
        'dataset_info': []
    }
    
    position_offset = 0
    
    for dataset_name in datasets:
        logger.info(f"\nProcessing {dataset_name}...")
        
        # Load activation differences 
        differences, tokens = load_activation_differences(dataset_name, split='train', max_shards=1)
        
        if differences is None:
            continue
        
        # Run SAE inference
        positions, indices, values = run_sae_on_activations(sae, differences, batch_size=512)
        
        # Adjust positions to global offset
        positions = [p + position_offset for p in positions]
        
        # Store results
        all_results['positions'].extend(positions)
        all_results['indices'].extend(indices) 
        all_results['values'].extend(values)
        all_results['dataset_info'].append({
            'dataset': dataset_name,
            'start_pos': position_offset,
            'end_pos': position_offset + differences.shape[0],
            'num_activations': len(values)
        })
        
        position_offset += differences.shape[0]
        
        # Save tokens if available
        if tokens is not None:
            torch.save(tokens, output_dir / f'tokens_{dataset_name}.pt')
        
        # Clean up memory
        del differences
        if tokens is not None:
            del tokens
        gc.collect()
        torch.cuda.empty_cache()
    
    # Convert to tensors and save
    logger.info("\nSaving results...")
    
    # Create indices tensor (position, feature_id)
    indices_tensor = torch.zeros((len(all_results['positions']), 2), dtype=torch.long)
    indices_tensor[:, 0] = torch.tensor(all_results['positions'])
    indices_tensor[:, 1] = torch.tensor(all_results['indices'])
    
    values_tensor = torch.tensor(all_results['values'])
    
    # Save
    torch.save(indices_tensor, output_dir / 'indices.pt')
    torch.save(values_tensor, output_dir / 'activations.pt')
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(all_results['dataset_info'], f, indent=2)
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("Summary:")
    logger.info(f"Total positions processed: {position_offset}")
    logger.info(f"Total feature activations: {len(all_results['values'])}")
    logger.info(f"Unique features that activated: {len(set(all_results['indices']))}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Ready to create max activating examples database!")

if __name__ == "__main__":
    main()