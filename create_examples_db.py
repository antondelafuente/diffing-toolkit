#!/usr/bin/env python
"""
Create examples.db for SAE visualization from existing latent activations
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')

import torch
from pathlib import Path
from src.utils.max_act_store import MaxActStore
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

def create_examples_db():
    # Configuration
    sae_dir = Path('/workspace/diffing-toolkit/storage/diffing_results/gemma3_1B/roman_concrete/sae_difference/layer_12/SAEdiff_ftb-gemma3_1B-roman_concrete-L12-s1-t100-k100-lr1e-04-x16')
    cache_dir = sae_dir / 'latent_activations'
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    
    # Load the cached latent activations
    print("Loading latent activations...")
    sequences = torch.load(cache_dir / 'sequences.pt', weights_only=False)
    indices = torch.load(cache_dir / 'indices.pt', weights_only=False)
    activations = torch.load(cache_dir / 'activations.pt', weights_only=False)
    lengths = torch.load(cache_dir / 'lengths.pt', weights_only=False)
    ranges = torch.load(cache_dir / 'ranges.pt', weights_only=False)
    max_activations = torch.load(cache_dir / 'max_activations.pt', weights_only=False)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Total activations: {len(activations)}")
    print(f"Max activations shape: {max_activations.shape}")
    
    # Create examples database
    db_path = cache_dir / 'examples.db'
    if db_path.exists():
        print(f"Removing existing database at {db_path}")
        db_path.unlink()
    
    print(f"Creating examples database at {db_path}")
    
    # Initialize MaxActStore
    num_features = 18432
    max_store = MaxActStore(
        db_path=db_path,
        tokenizer=tokenizer,
        max_examples=20,  # Store top 20 examples per feature
        storage_format='dense',
        per_dataset=False
    )
    
    # Group activations by feature
    print("Grouping activations by feature...")
    feature_examples = {}
    
    for seq_idx in tqdm(range(min(500, len(sequences))), desc="Processing sequences"):
        # Get sequence tokens
        seq_length = lengths[seq_idx].item()
        tokens = sequences[seq_idx][:seq_length]
        
        # Get activations for this sequence
        if seq_idx < len(ranges) - 1:
            start_idx = ranges[seq_idx].item()
            end_idx = ranges[seq_idx + 1].item()
        else:
            start_idx = ranges[seq_idx].item() 
            end_idx = len(indices)
        
        # Get indices and values for this sequence
        seq_indices = indices[start_idx:end_idx]
        seq_activations = activations[start_idx:end_idx]
        
        # Filter to get only this sequence's activations
        # Handle both 0-based and 1-based indexing
        mask = (seq_indices[:, 0] == seq_idx) | (seq_indices[:, 0] == seq_idx + 1)
        if not mask.any():
            continue
            
        seq_indices = seq_indices[mask]
        seq_activations = seq_activations[mask]
        
        # Group by feature
        for (_, pos, feat), val in zip(seq_indices, seq_activations):
            if feat >= num_features or pos >= seq_length:
                continue
            feat = int(feat)
            if feat not in feature_examples:
                feature_examples[feat] = []
            feature_examples[feat].append({
                'seq_idx': seq_idx,
                'tokens': tokens.cpu() if torch.is_tensor(tokens) else tokens,
                'position': int(pos),
                'activation': float(val),
                'seq_length': seq_length
            })
    
    print(f"Found examples for {len(feature_examples)} features")
    
    # Add top examples for each feature to the store
    print("Adding examples to database...")
    for feature_id in tqdm(sorted(feature_examples.keys())[:1000], desc="Adding features"):  # First 1000 features
        examples = feature_examples[feature_id]
        # Sort by activation strength
        examples = sorted(examples, key=lambda x: x['activation'], reverse=True)[:20]
        
        if not examples:
            continue
        
        # Prepare batch data
        batch_tokens = []
        batch_scores = []
        batch_token_scores = []
        
        for ex in examples:
            tokens = ex['tokens']
            if torch.is_tensor(tokens):
                tokens = tokens.numpy()
            batch_tokens.append(tokens)
            batch_scores.append(ex['activation'])
            
            # Create per-token scores (only the active position has non-zero score)
            token_scores = np.zeros(len(tokens))
            token_scores[ex['position']] = ex['activation']
            batch_token_scores.append(token_scores)
        
        # Convert to tensors
        batch_scores = torch.tensor(batch_scores)
        
        # Pad sequences to same length
        max_len = max(len(t) for t in batch_tokens)
        padded_tokens = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
        padded_token_scores = torch.zeros((len(batch_tokens), max_len))
        
        for i, (tokens, scores) in enumerate(zip(batch_tokens, batch_token_scores)):
            padded_tokens[i, :len(tokens)] = torch.tensor(tokens)
            padded_token_scores[i, :len(scores)] = torch.tensor(scores)
        
        # Add to store
        max_store.add_batch_examples(
            scores_per_example=batch_scores,
            input_ids_batch=padded_tokens,
            scores_per_token_batch=padded_token_scores,
            latent_idx=feature_id
        )
    
    # Finalize the database
    print("Finalizing database...")
    max_store.finalize()
    
    print(f"Examples database created successfully at {db_path}")
    print(f"Database size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verify it worked
    from src.utils.max_act_store import ReadOnlyMaxActStore
    ro_store = ReadOnlyMaxActStore(db_path, tokenizer=tokenizer)
    print(f"Verified: Database ready for visualization")
    
    # Test query
    examples = ro_store.get_examples(latent_idx=8301, k=5)
    if examples:
        print(f"\nTest: Found {len(examples)} examples for feature 8301")
    
if __name__ == "__main__":
    create_examples_db()