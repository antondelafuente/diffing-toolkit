#!/usr/bin/env python3
"""
Create examples.db for SAE visualization from the latent activations we just computed.
This builds a database of maximum activating examples for each SAE feature.
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import torch
from pathlib import Path
from src.utils.max_act_store import MaxActStore
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import json
from loguru import logger

def create_examples_db():
    # Configuration
    latent_dir = Path('/workspace/diffing-toolkit/storage/sae_latent_activations')
    output_dir = Path('/workspace/diffing-toolkit/storage/sae_dashboard')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer for Llama-3.2-1B-Instruct
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load the latent activations we just computed
    logger.info("Loading latent activations...")
    indices = torch.load(latent_dir / 'indices.pt', weights_only=True)
    activations = torch.load(latent_dir / 'activations.pt', weights_only=True)
    
    # Load dataset info to understand position ranges
    with open(latent_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    logger.info(f"Total activations: {len(activations)}")
    logger.info(f"Indices shape: {indices.shape}")
    
    # Load tokens for each dataset
    all_tokens = []
    position_to_dataset = {}
    
    for info in dataset_info:
        dataset_name = info['dataset']
        tokens_file = latent_dir / f'tokens_{dataset_name}.pt'
        
        if tokens_file.exists():
            tokens = torch.load(tokens_file, weights_only=True)
            logger.info(f"Loaded {len(tokens)} tokens for {dataset_name}")
            
            # Map positions to dataset and local offset
            for pos in range(info['start_pos'], info['end_pos']):
                position_to_dataset[pos] = {
                    'dataset': dataset_name,
                    'local_pos': pos - info['start_pos'],
                    'tokens': tokens
                }
        else:
            logger.warning(f"No tokens file found for {dataset_name}")
    
    # Create examples database
    db_path = output_dir / 'examples.db'
    if db_path.exists():
        logger.info(f"Removing existing database at {db_path}")
        db_path.unlink()
    
    logger.info(f"Creating examples database at {db_path}")
    
    # Initialize MaxActStore
    num_features = 4096  # SAE dict size
    max_store = MaxActStore(
        db_path=db_path,
        tokenizer=tokenizer,
        max_examples=50,  # Store top 50 examples per feature
        storage_format='dense',
        per_dataset=False
    )
    
    # Group activations by feature
    logger.info("Grouping activations by feature...")
    feature_examples = {}
    
    for i in tqdm(range(len(indices)), desc="Processing activations"):
        pos = indices[i, 0].item()
        feature_id = indices[i, 1].item()
        activation_value = activations[i].item()
        
        if feature_id not in feature_examples:
            feature_examples[feature_id] = []
        
        # Get the tokens for this position
        if pos in position_to_dataset:
            dataset_data = position_to_dataset[pos]
            local_pos = dataset_data['local_pos']
            tokens = dataset_data['tokens']
            
            # Handle the fact that tokens might be a flat list or need indexing
            # Each position represents a token in the flattened sequence
            if local_pos < len(tokens):
                # Create a context window around the active token
                context_size = 128  # Show 128 tokens of context
                start_idx = max(0, local_pos - context_size // 2)
                end_idx = min(len(tokens), start_idx + context_size)
                
                context_tokens = tokens[start_idx:end_idx]
                active_pos_in_context = local_pos - start_idx
                
                feature_examples[feature_id].append({
                    'tokens': context_tokens,
                    'position': active_pos_in_context,
                    'activation': activation_value,
                    'dataset': dataset_data['dataset']
                })
    
    logger.info(f"Found examples for {len(feature_examples)} features")
    
    # Add top examples for each feature to the store
    logger.info("Adding examples to database...")
    
    for feature_id in tqdm(sorted(feature_examples.keys()), desc="Adding features"):
        examples = feature_examples[feature_id]
        
        # Sort by activation strength and take top examples
        examples = sorted(examples, key=lambda x: abs(x['activation']), reverse=True)[:50]
        
        if not examples:
            continue
        
        # Prepare batch data
        batch_tokens = []
        batch_scores = []
        batch_token_scores = []
        
        for ex in examples:
            tokens = ex['tokens']
            if torch.is_tensor(tokens):
                tokens = tokens.cpu().numpy()
            else:
                tokens = np.array(tokens)
            
            batch_tokens.append(tokens)
            batch_scores.append(abs(ex['activation']))
            
            # Create per-token scores (only the active position has non-zero score)
            token_scores = np.zeros(len(tokens))
            if ex['position'] < len(tokens):
                token_scores[ex['position']] = abs(ex['activation'])
            batch_token_scores.append(token_scores)
        
        if not batch_tokens:
            continue
            
        # Convert to tensors
        batch_scores = torch.tensor(batch_scores)
        
        # Pad sequences to same length
        max_len = max(len(t) for t in batch_tokens)
        padded_tokens = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
        padded_token_scores = torch.zeros((len(batch_tokens), max_len))
        
        for i, (tokens, scores) in enumerate(zip(batch_tokens, batch_token_scores)):
            padded_tokens[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            padded_token_scores[i, :len(scores)] = torch.tensor(scores)
        
        # Add to store
        max_store.add_batch_examples(
            scores_per_example=batch_scores,
            input_ids_batch=padded_tokens,
            scores_per_token_batch=padded_token_scores,
            latent_idx=feature_id
        )
    
    # Finalize the database
    logger.info("Finalizing database...")
    max_store.finalize()
    
    logger.info(f"Examples database created successfully at {db_path}")
    logger.info(f"Database size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verify it worked
    from src.utils.max_act_store import ReadOnlyMaxActStore
    ro_store = ReadOnlyMaxActStore(db_path, tokenizer=tokenizer)
    logger.info(f"Verified: Database ready for visualization")
    
    # Test query - try a few different features
    test_features = [0, 100, 500, 1000, 2000]
    for feat_id in test_features:
        if feat_id in feature_examples:
            examples = ro_store.get_examples(latent_idx=feat_id, k=5)
            if examples:
                logger.info(f"Test: Found {len(examples)} examples for feature {feat_id}")
                break

if __name__ == "__main__":
    create_examples_db()