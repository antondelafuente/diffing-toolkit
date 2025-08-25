#!/usr/bin/env python3
"""
Build a comprehensive database mapping features to their maximum activating text.
Efficiently processes all available data to create a reusable lookup table.

Strategy:
1. Process data in streaming fashion to avoid memory issues
2. Use a priority queue per feature to keep only top-k examples
3. Save incrementally to avoid losing progress
4. Support resuming from interruption
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

print("Starting build_comprehensive_feature_db.py...", flush=True)

import torch
import numpy as np
from pathlib import Path
import json
import sqlite3
from collections import defaultdict
import heapq
from loguru import logger
from tqdm import tqdm
import gc
from datetime import datetime
import pickle

from dictionary_learning import BatchTopKSAE
from transformers import AutoTokenizer

class FeatureExampleDB:
    """Efficient database for storing feature -> max activating examples mapping."""
    
    def __init__(self, db_path: Path, max_examples_per_feature: int = 20):
        self.db_path = db_path
        self.max_examples = max_examples_per_feature
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Main examples table - LEAN version, no redundant data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_examples (
                feature_id INTEGER,
                rank INTEGER,
                activation_value REAL,
                dataset TEXT,
                global_position INTEGER,
                context_start INTEGER,
                context_end INTEGER,
                active_token_position INTEGER,
                PRIMARY KEY (feature_id, rank)
            )
        """)
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_activation 
            ON feature_examples(feature_id, activation_value DESC)
        """)
        
        self.conn.commit()
    
    def add_examples(self, feature_id: int, examples: list, tokenizer=None):
        """Add examples for a feature, keeping only top-k."""
        cursor = self.conn.cursor()
        
        # Get existing examples for this feature
        cursor.execute("""
            SELECT rank, activation_value FROM feature_examples 
            WHERE feature_id = ? 
            ORDER BY activation_value DESC
        """, (feature_id,))
        existing = cursor.fetchall()
        
        # Combine with new examples
        all_examples = []
        
        # Add existing (converting activation_value to float)
        for rank, act_val in existing:
            cursor.execute("""
                SELECT * FROM feature_examples 
                WHERE feature_id = ? AND rank = ?
            """, (feature_id, rank))
            row = cursor.fetchone()
            if row:
                # Ensure activation value is float (index 2)
                row_list = list(row)
                row_list[2] = float(row_list[2]) if row_list[2] is not None else 0.0
                all_examples.append(tuple(row_list))
        
        # Add new examples (as tuples matching DB schema)
        for ex in examples:
            text = ""
            if tokenizer and 'context_tokens' in ex:
                tokens = ex['context_tokens']
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.cpu().numpy()
                text = tokenizer.decode(tokens, skip_special_tokens=False)
            
            all_examples.append((
                feature_id,
                -1,  # rank will be reassigned
                float(ex['activation']),  # Ensure it's a Python float
                ex['dataset'],
                ex['global_position'],
                pickle.dumps(ex.get('context_tokens', [])),
                ex.get('active_token_position', -1),
                text
            ))
        
        # Sort by activation value and keep top-k
        all_examples.sort(key=lambda x: x[2], reverse=True)
        top_examples = all_examples[:self.max_examples]
        
        # Delete existing examples for this feature
        cursor.execute("DELETE FROM feature_examples WHERE feature_id = ?", (feature_id,))
        
        # Insert top examples with updated ranks
        for rank, ex in enumerate(top_examples):
            cursor.execute("""
                INSERT INTO feature_examples 
                (feature_id, rank, activation_value, dataset, global_position, 
                 context_tokens, active_token_position, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (feature_id, rank, ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]))
        
        self.conn.commit()
    
    def get_examples(self, feature_id: int, top_k: int = 10):
        """Get top-k examples for a feature."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM feature_examples 
            WHERE feature_id = ? 
            ORDER BY rank 
            LIMIT ?
        """, (feature_id, top_k))
        return cursor.fetchall()
    
    def get_features_with_examples(self):
        """Get list of all features that have examples."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT feature_id FROM feature_examples")
        return [row[0] for row in cursor.fetchall()]
    
    def save_metadata(self, key: str, value: str):
        """Save metadata."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)
        """, (key, value))
        self.conn.commit()
    
    def get_metadata(self, key: str):
        """Get metadata."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def add_examples_batch(self, feature_examples_dict, tokenizer=None):
        """Add examples for multiple features at once (much faster)."""
        cursor = self.conn.cursor()
        
        # Turn off auto-commit for speed
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        cursor.execute("BEGIN TRANSACTION")
        
        # For initial load (no existing data), just insert directly
        logger.info(f"Processing {len(feature_examples_dict)} features...")
        
        for feat_id, examples in feature_examples_dict.items():
            # Keep only top examples
            examples = sorted(examples, key=lambda x: x['activation'], reverse=True)[:self.max_examples]
            
            # Insert examples
            for rank, ex in enumerate(examples):
                text = ""
                if tokenizer and 'context_tokens' in ex:
                    tokens = ex['context_tokens']
                    if isinstance(tokens, torch.Tensor):
                        tokens = tokens.cpu().numpy()
                    if tokens is not None:
                        text = tokenizer.decode(tokens, skip_special_tokens=False)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO feature_examples 
                    (feature_id, rank, activation_value, dataset, global_position, 
                     context_tokens, active_token_position, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feat_id, rank, float(ex['activation']), ex['dataset'],
                    ex['global_position'], pickle.dumps(ex.get('context_tokens', [])),
                    ex.get('active_token_position', -1), text
                ))
        
        # Commit all changes at once
        cursor.execute("COMMIT")
        self.conn.commit()
        logger.info("Database batch update complete")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def process_shard_for_features(sae, base_memmap, ft_memmap, tokens, 
                               shard_idx, shard_size, actual_size, 
                               dataset_name, d_model):
    """
    Process a single shard and extract feature activations.
    Returns a dict mapping feature_id to list of examples.
    """
    device = next(sae.parameters()).device
    feature_examples = defaultdict(list)
    
    # Process in batches
    batch_size = 256
    num_batches = (actual_size + batch_size - 1) // batch_size
    logger.info(f"Processing shard {shard_idx} with {actual_size} positions in {num_batches} batches")
    
    for batch_idx, batch_start in enumerate(tqdm(range(0, actual_size, batch_size), 
                           desc=f"{dataset_name} shard {shard_idx}", leave=False)):
        batch_end = min(batch_start + batch_size, actual_size)
        
        # Log progress every 10 batches
        if batch_idx % 10 == 0:
            progress = (batch_idx / num_batches) * 100
            logger.info(f"  Shard {shard_idx} progress: {progress:.1f}% ({batch_idx}/{num_batches} batches)")
        
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
                activated_features = topk_idx[i].cpu().numpy()
                activated_values = latent_acts_dense[i, topk_idx[i]].cpu().numpy()
                
                # Get context tokens if available
                context = None
                active_pos = None
                if tokens is not None and global_pos < len(tokens):
                    context_start = max(0, global_pos - 15)
                    context_end = min(len(tokens), global_pos + 16)
                    context = tokens[context_start:context_end]
                    active_pos = global_pos - context_start
                
                # Store activation for each feature
                for feat_idx, feat_val in zip(activated_features, activated_values):
                    if abs(feat_val) > 0.1:  # Filter very small activations
                        feature_examples[feat_idx].append({
                            'activation': float(abs(feat_val)),  # Convert to Python float
                            'dataset': dataset_name,
                            'global_position': global_pos,
                            'context_tokens': context,
                            'active_token_position': active_pos
                        })
        
        # Clean up memory periodically
        if batch_start % 2048 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return feature_examples


def build_comprehensive_db():
    """Build comprehensive feature database from all available data."""
    
    print("Entering build_comprehensive_db()", flush=True)
    logger.info("="*60)
    logger.info("Building Comprehensive Feature Database")
    logger.info("="*60)
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    print("Creating database...", flush=True)
    db_path = Path('/workspace/diffing-toolkit/comprehensive_feature_db.sqlite')
    db = FeatureExampleDB(db_path, max_examples_per_feature=20)
    
    # Check if resuming
    last_checkpoint = db.get_metadata('last_checkpoint')
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        last_dataset, last_shard = last_checkpoint.split(':')
        last_shard = int(last_shard)
        resume = True
    else:
        resume = False
        last_dataset = None
        last_shard = -1
    
    # Load SAE
    print("About to load SAE...", flush=True)
    logger.info("Loading 12k SAE...")
    checkpoint_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    print(f"Loading from {checkpoint_path}", flush=True)
    sae = BatchTopKSAE.from_pretrained(checkpoint_path)
    print("SAE loaded, setting to eval mode...", flush=True)
    sae.eval()
    sae = sae.cuda()
    print("SAE moved to CUDA", flush=True)
    
    # Process each dataset
    datasets = [
        ('bad_medical_advice.jsonl', 8),   # All 8 shards
        ('tulu-3-sft-olmo-2-mixture', 10), # First 10 shards (reasonable sample)
        ('fineweb-1m-sample', 10),         # First 10 shards (reasonable sample)
    ]
    
    base_path = Path('/workspace/diffing-toolkit/storage/activations_merged_custom')
    
    for dataset_name, num_shards in datasets:
        # Skip if already processed
        if resume and dataset_name < last_dataset:
            logger.info(f"Skipping {dataset_name} (already processed)")
            continue
            
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
            tokens = torch.load(tokens_path, map_location='cpu')
            logger.info(f"Loaded tokens: {tokens.shape}")
        else:
            tokens = None
        
        # Process each shard
        for shard_idx in range(num_shards):
            # Skip if already processed
            if resume and dataset_name == last_dataset and shard_idx <= last_shard:
                logger.info(f"Skipping shard {shard_idx} (already processed)")
                continue
            
            shard_start_time = datetime.now()
            logger.info(f"\nProcessing shard {shard_idx}/{num_shards-1}...")
            logger.info(f"Time elapsed: {(datetime.now() - start_time).total_seconds():.1f}s")
            
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
            
            # Process shard
            feature_examples = process_shard_for_features(
                sae, base_memmap, ft_memmap, tokens,
                shard_idx, shard_size, actual_size,
                dataset_name, d_model
            )
            
            # Add to database
            num_features = len(feature_examples)
            num_examples = sum(len(ex) for ex in feature_examples.values())
            logger.info(f"Shard {shard_idx} complete: {num_features} features, {num_examples} total examples")
            logger.info(f"Adding to database (batch mode)...")
            
            # Batch process all features at once
            db.add_examples_batch(feature_examples, tokenizer)
            
            # Save checkpoint
            db.save_metadata('last_checkpoint', f'{dataset_name}:{shard_idx}')
            db.save_metadata('last_update', datetime.now().isoformat())
            logger.info(f"Checkpoint saved: {dataset_name}:{shard_idx}")
            
            # Clean up memory
            del base_memmap, ft_memmap, feature_examples
            gc.collect()
    
    # Final metadata
    db.save_metadata('completed', 'true')
    db.save_metadata('completion_time', datetime.now().isoformat())
    db.save_metadata('num_features', str(len(db.get_features_with_examples())))
    
    logger.info("\n" + "="*60)
    logger.info("Database building complete!")
    logger.info(f"Database saved to: {db_path}")
    logger.info(f"Features with examples: {len(db.get_features_with_examples())}")
    
    db.close()


if __name__ == "__main__":
    build_comprehensive_db()