#!/usr/bin/env python3
"""
Quick version: Create examples.db focusing on top 100 features with 10 examples each.
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
from collections import defaultdict

def create_examples_db():
    # Configuration
    latent_dir = Path('/workspace/diffing-toolkit/storage/sae_latent_activations')
    output_dir = Path('/workspace/diffing-toolkit/storage/sae_dashboard')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer for Llama-3.2-1B-Instruct
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load the latent activations
    logger.info("Loading latent activations...")
    indices = torch.load(latent_dir / 'indices.pt', weights_only=True)
    activations = torch.load(latent_dir / 'activations.pt', weights_only=True)
    
    # Load dataset info
    with open(latent_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    logger.info(f"Total activations: {len(activations)}")
    
    # Load tokens for each dataset
    tokens_by_dataset = {}
    for info in dataset_info:
        dataset_name = info['dataset']
        tokens_file = latent_dir / f'tokens_{dataset_name}.pt'
        if tokens_file.exists():
            tokens = torch.load(tokens_file, weights_only=True)
            tokens_by_dataset[dataset_name] = {
                'tokens': tokens,
                'start_pos': info['start_pos'],
                'end_pos': info['end_pos']
            }
            logger.info(f"Loaded {len(tokens)} tokens for {dataset_name}")
    
    # Find top features by total activation magnitude
    logger.info("Finding top features by activation strength...")
    feature_totals = defaultdict(float)
    feature_examples = defaultdict(list)
    
    for i in range(len(indices)):
        pos = indices[i, 0].item()
        feature_id = indices[i, 1].item()
        activation_value = abs(activations[i].item())
        
        feature_totals[feature_id] += activation_value
        
        # Find which dataset this position belongs to
        for dataset_name, data in tokens_by_dataset.items():
            if data['start_pos'] <= pos < data['end_pos']:
                local_pos = pos - data['start_pos']
                tokens = data['tokens']
                
                if local_pos < len(tokens):
                    # Store example info
                    feature_examples[feature_id].append({
                        'activation': activation_value,
                        'position': local_pos,
                        'tokens': tokens,
                        'dataset': dataset_name
                    })
                break
    
    # Get top 100 features by total activation
    top_features = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)[:100]
    logger.info(f"Selected top {len(top_features)} features")
    
    # Create database
    db_path = output_dir / 'examples.db'
    if db_path.exists():
        logger.info(f"Removing existing database at {db_path}")
        db_path.unlink()
    
    logger.info(f"Creating examples database at {db_path}")
    
    # Initialize MaxActStore
    max_store = MaxActStore(
        db_path=db_path,
        tokenizer=tokenizer,
        max_examples=10,  # Only 10 examples per feature
        storage_format='dense',
        per_dataset=False
    )
    
    # Add examples for top features
    logger.info("Adding examples to database...")
    
    for feature_id, total_activation in tqdm(top_features, desc="Adding features"):
        examples = feature_examples[feature_id]
        
        # Sort by activation strength and take top 10
        examples = sorted(examples, key=lambda x: x['activation'], reverse=True)[:10]
        
        if not examples:
            continue
        
        # Prepare batch data
        batch_tokens = []
        batch_scores = []
        batch_token_scores = []
        
        for ex in examples:
            # Create a context window
            local_pos = ex['position']
            tokens = ex['tokens']
            
            context_size = 64  # Smaller context window
            start_idx = max(0, local_pos - context_size // 2)
            end_idx = min(len(tokens), start_idx + context_size)
            
            context_tokens = tokens[start_idx:end_idx]
            active_pos_in_context = local_pos - start_idx
            
            if torch.is_tensor(context_tokens):
                context_tokens = context_tokens.cpu().numpy()
            else:
                context_tokens = np.array(context_tokens)
            
            batch_tokens.append(context_tokens)
            batch_scores.append(ex['activation'])
            
            # Create per-token scores
            token_scores = np.zeros(len(context_tokens))
            if active_pos_in_context >= 0 and active_pos_in_context < len(context_tokens):
                token_scores[active_pos_in_context] = ex['activation']
            batch_token_scores.append(token_scores)
        
        if not batch_tokens:
            continue
        
        # Convert to tensors
        batch_scores = torch.tensor(batch_scores)
        
        # Pad sequences
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
    
    # Database is automatically saved
    logger.info(f"Examples database created successfully at {db_path}")
    logger.info(f"Database size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verify database exists
    if db_path.exists():
        logger.info(f"Verified: Database created at {db_path}")
        logger.info(f"Top feature IDs: {[f[0] for f in top_features[:5]]}")
    
    logger.info("Database ready for visualization!")

if __name__ == "__main__":
    create_examples_db()