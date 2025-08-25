#!/usr/bin/env python
"""
Create examples.db for 12k SAE visualization from latent activations and tokens
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')

import torch
from pathlib import Path
from src.utils.max_act_store import MaxActStore
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import json

def create_examples_db_12k():
    # Load the 12k SAE inference results
    cache_dir = Path('/workspace/diffing-toolkit/storage/sae_12k_latent_activations')
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load the cached latent activations
    print("Loading latent activations...")
    indices = torch.load(cache_dir / 'indices.pt', weights_only=False)
    activations = torch.load(cache_dir / 'activations.pt', weights_only=False)
    
    with open(cache_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    print(f"Total activations: {len(activations)}")
    print(f"Total positions: {indices[:, 0].max().item() + 1}")
    print(f"Datasets: {[d['dataset'] for d in dataset_info]}")
    
    # Create examples database
    db_path = cache_dir / 'examples.db'
    if db_path.exists():
        print(f"Removing existing database at {db_path}")
        db_path.unlink()
    
    print(f"Creating examples database at {db_path}")
    
    # Initialize MaxActStore
    num_features = 12288  # 12k features
    max_store = MaxActStore(
        db_path=db_path,
        tokenizer=tokenizer,
        max_examples=10,  # Store top 10 examples per feature
        storage_format='dense',
        per_dataset=False
    )
    
    # Load tokens for each dataset
    print("Loading tokens...")
    all_tokens = {}
    for dataset in ['bad_medical_advice.jsonl', 'tulu-3-sft-olmo-2-mixture', 'fineweb-1m-sample']:
        token_file = cache_dir / f'tokens_{dataset}.pt'
        if token_file.exists():
            tokens = torch.load(token_file, weights_only=False)
            all_tokens[dataset] = tokens
            print(f"  {dataset}: {tokens.shape}")
    
    # Group activations by feature
    print("Grouping activations by feature...")
    feature_examples = {}
    
    # Process activations
    for dataset_info_item in dataset_info:
        dataset_name = dataset_info_item['dataset']
        start_pos = dataset_info_item['start_pos']
        end_pos = dataset_info_item['end_pos']
        
        print(f"Processing {dataset_name} (positions {start_pos}-{end_pos})")
        
        # Get tokens for this dataset
        if dataset_name not in all_tokens:
            print(f"  Warning: No tokens found for {dataset_name}")
            continue
            
        dataset_tokens = all_tokens[dataset_name]
        
        # Filter activations for this dataset
        mask = (indices[:, 0] >= start_pos) & (indices[:, 0] < end_pos)
        dataset_indices = indices[mask]
        dataset_activations = activations[mask]
        
        print(f"  Found {len(dataset_activations)} activations")
        
        # Process each activation
        for (pos, feat), val in tqdm(zip(dataset_indices, dataset_activations), 
                                   desc=f"Processing {dataset_name}", leave=False):
            pos = pos.item()
            feat = feat.item()
            val = val.item()
            
            if feat >= num_features:
                continue
                
            # Calculate token position within this dataset
            token_pos = pos - start_pos
            if token_pos >= len(dataset_tokens):
                continue
            
            # Get context window around the token
            context_start = max(0, token_pos - 10)
            context_end = min(len(dataset_tokens), token_pos + 11)
            context_tokens = dataset_tokens[context_start:context_end]
            relative_pos = token_pos - context_start
            
            if feat not in feature_examples:
                feature_examples[feat] = []
                
            feature_examples[feat].append({
                'tokens': context_tokens,
                'position': relative_pos,
                'activation': val,
                'dataset': dataset_name,
                'global_pos': pos
            })
    
    print(f"Found examples for {len(feature_examples)} features")
    
    # Add top examples for each feature to the store
    print("Adding examples to database...")
    
    # Sort features by number of examples (most active first)
    sorted_features = sorted(feature_examples.keys(), 
                           key=lambda f: len(feature_examples[f]), reverse=True)
    
    for feature_id in tqdm(sorted_features[:2000], desc="Adding features"):  # Top 2000 most active features
        examples = feature_examples[feature_id]
        # Sort by activation strength
        examples = sorted(examples, key=lambda x: x['activation'], reverse=True)[:10]
        
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
            if ex['position'] < len(token_scores):
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
    
    # Test query - find a feature that should have examples
    test_features = sorted_features[:10]
    for test_feat in test_features:
        examples = ro_store.get_examples(latent_idx=test_feat, k=5)
        if examples:
            print(f"\nTest: Found {len(examples)} examples for feature {test_feat}")
            print(f"  Max activation: {examples[0]['score']:.3f}")
            break
    
    return db_path

if __name__ == "__main__":
    create_examples_db_12k()