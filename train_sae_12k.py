#!/usr/bin/env python3
"""
Train SAE with 6x expansion (12,288 features) and k=48 sparsity
"""

import sys
import os
sys.path.append('/workspace/diffing-toolkit')
sys.path.append('/workspace/diffing-toolkit/.local')

from pathlib import Path
import torch
import numpy as np
from dictionary_learning import BatchTopKSAE
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import wandb
from datetime import datetime

def main():
    # Configuration
    dict_size = 12288  # 6x expansion
    k = 48  # Very sparse
    activation_dim = 2048
    batch_size = 2048
    learning_rate = 1e-4
    num_epochs = 3
    num_samples = 15_600_000
    
    # Paths
    base_path = Path('/workspace/diffing-toolkit/storage/activations_merged_custom')
    checkpoint_dir = Path('/workspace/diffing-toolkit/storage/checkpoints_12k')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb (disable for now - no API key)
    use_wandb = False
    if use_wandb:
        wandb.init(
            project="Diffing-Game-DiffSAE",
            name=f"SAE_12k_k48_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "dict_size": dict_size,
                "k": k,
                "activation_dim": activation_dim,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "num_samples": num_samples,
                "expansion_factor": 6,
            }
        )
    
    logger.info("="*60)
    logger.info("SAE Training Configuration")
    logger.info(f"Dictionary size: {dict_size} (6x expansion)")
    logger.info(f"Sparsity k: {k} ({k/dict_size*100:.2f}% active)")
    logger.info(f"Training samples: {num_samples:,}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Total training tokens: {num_samples * num_epochs:,}")
    logger.info("="*60)
    
    # Initialize SAE
    sae = BatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=dict_size,
        k=k
    )
    sae = sae.cuda()  # Move to GPU after initialization
    
    # Load activation differences
    logger.info("Loading activation differences...")
    
    all_differences = []
    datasets = [
        ('bad_medical_advice.jsonl', 8),  # 8 shards
        ('tulu-3-sft-olmo-2-mixture', 36),  # 36 shards  
        ('fineweb-1m-sample', 100)  # 100 shards (but merged)
    ]
    
    for dataset_name, num_shards in datasets:
        logger.info(f"Loading {dataset_name}...")
        
        base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / "train" / "layer_7_out"
        ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / "train" / "layer_7_out"
        
        if not base_dir.exists() or not ft_dir.exists():
            logger.warning(f"Skipping {dataset_name} - directories don't exist")
            continue
        
        # Load config to get dimensions
        import json
        with open(base_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        d_model = config['d_model']
        shard_count = min(config['shard_count'], num_shards)
        
        # Load from each shard
        for shard_idx in range(shard_count):
            if shard_idx >= 1 and dataset_name == 'bad_medical_advice.jsonl':
                continue  # Only use first shard for testing
                
            logger.info(f"  Loading shard {shard_idx}/{shard_count-1}")
            
            # Load memmap files
            base_memmap = np.memmap(
                base_dir / f'shard_{shard_idx}.memmap',
                dtype='float32',
                mode='r',
                shape=(config['shard_size'] // d_model, d_model)
            )
            
            ft_memmap = np.memmap(
                ft_dir / f'shard_{shard_idx}.memmap',
                dtype='float32',
                mode='r',
                shape=(config['shard_size'] // d_model, d_model)
            )
            
            # Calculate actual size
            if shard_idx == config['shard_count'] - 1:
                actual_size = config['total_size'] - shard_idx * (config['shard_size'] // d_model)
            else:
                actual_size = config['shard_size'] // d_model
            
            # Compute differences
            base_acts = torch.from_numpy(base_memmap[:actual_size]).float()
            ft_acts = torch.from_numpy(ft_memmap[:actual_size]).float()
            differences = ft_acts - base_acts
            
            all_differences.append(differences)
            
            # Clean up memory
            del base_memmap, ft_memmap, base_acts, ft_acts
            
            # Limit data for memory
            if len(all_differences) > 20:  # Process in chunks
                break
    
    # Concatenate all differences
    all_differences = torch.cat(all_differences, dim=0)
    logger.info(f"Total activation differences loaded: {all_differences.shape}")
    
    # Create dataloader
    dataset = TensorDataset(all_differences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info("Starting training...")
    step = 0
    save_every = 5000
    log_every = 100
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        epoch_auxk_loss = 0
        epoch_l1_loss = 0
        
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.cuda()
            
            # Forward pass
            loss, info = sae.forward(batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            if 'auxk_loss' in info:
                epoch_auxk_loss += info['auxk_loss'].item()
            if 'l1_loss' in info:
                epoch_l1_loss += info['l1_loss'].item()
            
            step += 1
            
            if step % log_every == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                logger.info(f"Step {step}, Loss: {avg_loss:.4f}")
                
                if wandb.run is not None:
                    wandb.log({
                        "loss": loss.item(),
                        "auxk_loss": info.get('auxk_loss', 0),
                        "l1_loss": info.get('l1_loss', 0),
                        "step": step,
                        "epoch": epoch
                    })
            
            if step % save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
                torch.save({
                    'model_state_dict': sae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'loss': loss.item()
                }, checkpoint_path)
                logger.info(f"Checkpoint saved at step {step}")
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_path = checkpoint_dir / "final_model"
    sae.save_to_disk(final_path)
    logger.info(f"Final model saved to {final_path}")
    
    # Upload to HuggingFace
    logger.info("Uploading to HuggingFace...")
    hf_repo = f"matonski/SAEdiff_llama32_1B_medical_12k_k48_x6"
    
    try:
        # Set HF token
        token_file = Path("/workspace/.hf_token")
        if token_file.exists():
            os.environ["HF_TOKEN"] = token_file.read_text().strip()
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token_file.read_text().strip()
        
        sae.save_to_hub(hf_repo)
        logger.info(f"Successfully uploaded to {hf_repo}")
    except Exception as e:
        logger.warning(f"Failed to upload to HuggingFace: {e}")
        logger.info("Model saved locally, can upload manually later")
    
    if use_wandb:
        wandb.finish()
    logger.info("Training complete!")

if __name__ == "__main__":
    main()