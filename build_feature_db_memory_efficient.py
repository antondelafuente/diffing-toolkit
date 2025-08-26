#!/usr/bin/env python3
"""
Memory-efficient feature database builder that keeps only top-N examples per feature.
Processes all features (no top-k constraint) but maintains constant memory usage.
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
import heapq

from dictionary_learning import BatchTopKSAE
from transformers import AutoTokenizer
from src.utils.max_act_store import MaxActStore


class TopKTracker:
    """Efficiently tracks top-k examples for each feature using heaps."""
    
    def __init__(self, k=20):
        self.k = k
        # Min-heap for each feature - keeps top k highest values
        # Heap contains tuples of (score, seq_idx, position)
        self.feature_heaps = defaultdict(list)
    
    def add(self, feature_id, score, seq_idx, position):
        """Add a new example, keeping only top-k."""
        heap = self.feature_heaps[feature_id]
        
        if len(heap) < self.k:
            # Haven't reached k examples yet, just add
            heapq.heappush(heap, (score, seq_idx, position))
        elif score > heap[0][0]:
            # New score is better than worst in heap
            heapq.heapreplace(heap, (score, seq_idx, position))
    
    def get_top_examples(self):
        """Get all top examples, sorted by score."""
        result = {}
        for feature_id, heap in self.feature_heaps.items():
            # Sort by score (descending)
            sorted_examples = sorted(heap, key=lambda x: x[0], reverse=True)
            result[feature_id] = sorted_examples
        return result


def process_dataset_for_features(sae, dataset_name, num_shards, top_k=20):
    """
    Process a dataset and collect top-k feature activations.
    Memory-efficient version that maintains constant memory usage.
    """
    base_path = Path('/workspace/diffing-toolkit/storage/activations_merged_custom')
    base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / "train" / "layer_7_out"
    ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / "train" / "layer_7_out"
    
    if not base_dir.exists() or not ft_dir.exists():
        logger.warning(f"Skipping {dataset_name} - directories don't exist")
        return TopKTracker(top_k), {}
    
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
        return TopKTracker(top_k), {}
    
    device = next(sae.parameters()).device
    
    # Initialize trackers
    tracker = TopKTracker(top_k)
    sequences = {}  # seq_idx -> tokens
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
                
                # Process each position WITHOUT top-k filtering
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
                        
                        # Process ALL features (not just top-k)
                        all_feature_values = latent_acts_dense[i].cpu().numpy()
                        
                        # Calculate which token in the window is the active one
                        active_position_in_window = global_pos - context_start
                        
                        # Add to tracker (only keeps top-k internally)
                        for feat_idx in range(len(all_feature_values)):
                            feat_val = all_feature_values[feat_idx]
                            if abs(feat_val) > 0:  # Filter only true zeros
                                tracker.add(
                                    feat_idx,
                                    float(abs(feat_val)),
                                    seq_idx,
                                    active_position_in_window
                                )
        
        # Clean up memory
        del base_memmap, ft_memmap
        gc.collect()
        
        # Log progress with memory usage
        if shard_idx % 5 == 0:
            num_features_with_examples = len(tracker.feature_heaps)
            import psutil
            process = psutil.Process()
            mem_usage = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            logger.info(f"  Shard {shard_idx}: {len(sequences)} sequences, "
                       f"{num_features_with_examples} features with examples, "
                       f"Memory: {mem_usage:.1f} GB")
    
    return tracker, sequences


def build_efficient_db():
    """Build efficient feature database with constant memory usage."""
    
    logger.info("="*60)
    logger.info("Memory-Efficient Feature Database Builder")
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
        ('bad_medical_advice.jsonl', 8),    # All 8 shards (0-7)
        ('tulu-3-sft-olmo-2-mixture', 36),  # All 36 shards (0-35)
        ('fineweb-1m-sample', 100),         # All 100 shards (0-99)
    ]
    
    # Global trackers
    global_tracker = TopKTracker(k=20)
    all_sequences = {}
    
    for dataset_name, num_shards in datasets:
        logger.info(f"\nProcessing {dataset_name} ({num_shards} shards)...")
        
        tracker, sequences = process_dataset_for_features(sae, dataset_name, num_shards, top_k=100)
        
        # Merge sequences
        seq_idx_offset = len(all_sequences)
        for seq_idx, tokens in sequences.items():
            all_sequences[seq_idx + seq_idx_offset] = tokens
        
        # Merge top examples into global tracker
        dataset_top = tracker.get_top_examples()
        for feat_idx, examples in dataset_top.items():
            for score, seq_idx, active_pos in examples:
                global_tracker.add(feat_idx, score, seq_idx + seq_idx_offset, active_pos)
        
        logger.info(f"  Total: {len(all_sequences)} sequences, {len(global_tracker.feature_heaps)} features with examples")
        
        # Clean up dataset-specific data
        del tracker
        gc.collect()
    
    # Convert to final format
    logger.info("\nPreparing final database...")
    quantile_examples = {0: {}}  # Single quantile for simplicity
    active_positions = {}  # Store active positions separately
    
    final_examples = global_tracker.get_top_examples()
    for feat_idx, examples in tqdm(final_examples.items(), desc="Formatting"):
        # Extract just (score, seq_idx) for MaxActStore
        quantile_examples[0][feat_idx] = [(score, seq_idx) for score, seq_idx, _ in examples]
        # Store active positions separately
        active_positions[feat_idx] = [(seq_idx, pos) for _, seq_idx, pos in examples]
    
    # Convert sequences to list format
    sequences_list = list(all_sequences.items())
    
    # Save using MaxActStore
    db_path = Path('/workspace/diffing-toolkit/efficient_feature_db_memory_safe')
    db_path.mkdir(exist_ok=True)
    
    logger.info(f"\nSaving to MaxActStore database...")
    max_store = MaxActStore(db_path / "examples.db", tokenizer=tokenizer)
    
    max_store.fill(
        examples_data=quantile_examples,
        all_sequences=sequences_list,
        activation_details=None,  # We don't have detailed activations
        dataset_info=None
    )
    
    # Save active positions separately
    positions_path = db_path / "active_positions.json"
    with open(positions_path, 'w') as f:
        # Convert to JSON-serializable format
        positions_data = {
            str(feat_id): [(int(seq), int(pos)) for seq, pos in positions]
            for feat_id, positions in active_positions.items()
        }
        json.dump(positions_data, f)
    logger.info(f"Active positions saved to: {positions_path}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("="*60)
    logger.info(f"Database building complete in {elapsed:.1f} seconds!")
    logger.info(f"Database saved to: {db_path}")
    logger.info(f"Features with examples: {len(quantile_examples[0])}")
    logger.info(f"Total unique sequences: {len(sequences_list)}")
    
    # Show final memory usage
    import psutil
    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    logger.info(f"Final memory usage: {mem_usage:.1f} GB")


if __name__ == "__main__":
    build_efficient_db()