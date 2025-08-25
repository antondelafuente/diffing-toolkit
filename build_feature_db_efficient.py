#!/usr/bin/env python3
"""
Efficient feature database builder that mimics MaxActStore's approach.
Stores sequences once and uses references to avoid redundancy.
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import torch
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import gc
from datetime import datetime

from dictionary_learning import BatchTopKSAE
from transformers import AutoTokenizer
from src.utils.max_act_store import MaxActStore

def process_dataset_for_features(sae, dataset_name, num_shards):
    """
    Process a dataset and collect feature activations.
    Returns sequences and feature examples in MaxActStore format.
    """
    base_path = Path('/workspace/diffing-toolkit/storage/activations_merged_custom')
    base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / "train" / "layer_7_out"
    ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / "train" / "layer_7_out"
    
    if not base_dir.exists() or not ft_dir.exists():
        logger.warning(f"Skipping {dataset_name} - directories don't exist")
        return {}, []
    
    # Load config
    with open(base_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    d_model = config['d_model']
    shard_size = config['shard_size'] // d_model
    
    # Load tokens
    tokens_path = base_dir.parent / 'tokens.pt'
    if tokens_path.exists():
        all_tokens = torch.load(tokens_path, map_location='cpu')
        logger.info(f"Loaded tokens for {dataset_name}: {all_tokens.shape}")
    else:
        all_tokens = None
        logger.warning(f"No tokens found for {dataset_name}")
        return {}, []
    
    device = next(sae.parameters()).device
    
    # Collect unique sequences and feature examples
    sequences = {}  # seq_idx -> tokens
    feature_examples = defaultdict(list)  # feature_id -> [(score, seq_idx)]
    next_seq_idx = 0
    
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
            
            # Run through SAE
            with torch.no_grad():
                latent_acts_dense = sae.encode(differences)
                
                # Apply top-k sparsity
                k = sae.k
                topk_vals, topk_idx = torch.topk(latent_acts_dense.abs(), k=k, dim=1)
                
                # Process each position
                for i in range(differences.shape[0]):
                    global_pos = shard_idx * shard_size + batch_start + i
                    
                    # Get context tokens (30 tokens window)
                    if all_tokens is not None and global_pos < len(all_tokens):
                        context_start = max(0, global_pos - 15)
                        context_end = min(len(all_tokens), global_pos + 16)
                        context_tokens = all_tokens[context_start:context_end]
                        
                        # Create a tuple key for deduplication
                        tokens_key = tuple(context_tokens.tolist())
                        
                        # Check if we've seen this sequence before
                        if tokens_key not in sequences:
                            sequences[next_seq_idx] = context_tokens
                            seq_idx = next_seq_idx
                            next_seq_idx += 1
                        else:
                            # Find existing sequence index
                            seq_idx = None
                            for idx, seq_tokens in sequences.items():
                                if tuple(seq_tokens.tolist()) == tokens_key:
                                    seq_idx = idx
                                    break
                        
                        # Store activations for each feature
                        activated_features = topk_idx[i].cpu().numpy()
                        activated_values = latent_acts_dense[i, topk_idx[i]].cpu().numpy()
                        
                        for feat_idx, feat_val in zip(activated_features, activated_values):
                            if abs(feat_val) > 0.1:  # Filter very small activations
                                feature_examples[int(feat_idx)].append((
                                    float(abs(feat_val)),  # score
                                    seq_idx  # sequence index
                                ))
        
        # Clean up memory
        del base_memmap, ft_memmap
        gc.collect()
        
        # Log progress
        logger.info(f"  Shard {shard_idx}: {len(sequences)} unique sequences, "
                   f"{len(feature_examples)} features with examples")
    
    return feature_examples, list(sequences.items())


def build_efficient_db():
    """Build efficient feature database using MaxActStore format."""
    
    logger.info("="*60)
    logger.info("Building Efficient Feature Database")
    logger.info("="*60)
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load SAE
    logger.info("Loading 12k SAE...")
    checkpoint_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    sae = BatchTopKSAE.from_pretrained(checkpoint_path)
    sae.eval()
    sae = sae.cuda()
    
    # Process each dataset
    datasets = [
        ('bad_medical_advice.jsonl', 8),   # All 8 shards
        ('tulu-3-sft-olmo-2-mixture', 10), # 10 shards
        ('fineweb-1m-sample', 10),         # 10 shards
    ]
    
    all_sequences = []
    all_feature_examples = defaultdict(list)
    
    for dataset_name, num_shards in datasets:
        logger.info(f"\nProcessing {dataset_name} ({num_shards} shards)...")
        
        feature_examples, sequences = process_dataset_for_features(sae, dataset_name, num_shards)
        
        # Merge sequences (with new indices)
        seq_idx_offset = len(all_sequences)
        for seq_idx, tokens in sequences:
            all_sequences.append((seq_idx + seq_idx_offset, tokens))
        
        # Merge feature examples (adjusting sequence indices)
        for feat_idx, examples in feature_examples.items():
            for score, seq_idx in examples:
                all_feature_examples[feat_idx].append((score, seq_idx + seq_idx_offset))
        
        logger.info(f"  Total: {len(all_sequences)} sequences, {len(all_feature_examples)} features")
    
    # Keep only top-k examples per feature
    logger.info("\nFiltering to top-20 examples per feature...")
    quantile_examples = {0: {}}  # Single quantile for simplicity
    
    for feat_idx, examples in tqdm(all_feature_examples.items(), desc="Filtering"):
        # Sort by score and keep top 20
        sorted_examples = sorted(examples, key=lambda x: x[0], reverse=True)[:20]
        quantile_examples[0][feat_idx] = sorted_examples
    
    # Save using MaxActStore
    db_path = Path('/workspace/diffing-toolkit/efficient_feature_db')
    db_path.mkdir(exist_ok=True)
    
    logger.info(f"\nSaving to MaxActStore database...")
    max_store = MaxActStore(db_path / "examples.db", tokenizer=tokenizer)
    
    max_store.fill(
        examples_data=quantile_examples,
        all_sequences=all_sequences,
        activation_details=None,  # We don't have detailed activations
        dataset_info=None
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("="*60)
    logger.info(f"Database building complete in {elapsed:.1f} seconds!")
    logger.info(f"Database saved to: {db_path}")
    logger.info(f"Features with examples: {len(quantile_examples[0])}")
    logger.info(f"Total unique sequences: {len(all_sequences)}")
    
    max_store.close()


if __name__ == "__main__":
    build_efficient_db()