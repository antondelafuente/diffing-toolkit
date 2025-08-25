#!/usr/bin/env python3
"""
Upload the best SAE checkpoint to HuggingFace Hub.
"""

from pathlib import Path
import torch
import json
from loguru import logger
import sys
import os

# Add project to path
sys.path.append('/workspace/diffing-toolkit')
sys.path.append('/workspace/diffing-toolkit/.local')

# Set HF token
with open('/workspace/.hf_token', 'r') as f:
    os.environ['HF_TOKEN'] = f.read().strip()

from src.utils.dictionary.utils import push_dictionary_model, push_config_to_hub
from src.utils.configs import HF_NAME

def main():
    logger.info("=== Uploading Best SAE Checkpoint to HuggingFace ===")
    
    # Configuration
    checkpoint_dir = Path("/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k100-lr1e-04-x2")
    best_checkpoint = "checkpoint_15000.pt"
    
    # First, copy the best checkpoint to model_final.pt (push_dictionary_model expects this name)
    best_path = checkpoint_dir / best_checkpoint
    final_path = checkpoint_dir / "model_final.pt"
    
    logger.info(f"Using best checkpoint: {best_checkpoint} (from validation)")
    logger.info(f"Copying {best_checkpoint} to model_final.pt for upload")
    
    # Copy the checkpoint
    import shutil
    shutil.copy2(best_path, final_path)
    
    # Update the config to reflect this is the best checkpoint
    config_path = checkpoint_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add metadata about which checkpoint was selected
    config['best_checkpoint'] = best_checkpoint
    config['best_checkpoint_step'] = 15000
    config['best_validation_variance_explained'] = 0.4727
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Updated config with best checkpoint info")
    
    # Now upload to HuggingFace
    try:
        logger.info("Uploading model to HuggingFace Hub...")
        hf_repo_id = push_dictionary_model(final_path)
        logger.info(f"✓ Successfully uploaded to: {hf_repo_id}")
        
        # Also save the HF repo ID for later use
        with open(checkpoint_dir / "hf_repo_id.txt", "w") as f:
            f.write(hf_repo_id)
        
        logger.info(f"Saved HF repo ID to {checkpoint_dir / 'hf_repo_id.txt'}")
        
        # Try to push config as well
        try:
            from omegaconf import OmegaConf
            # Create a minimal config for upload
            cfg = OmegaConf.create({
                "diffing": {
                    "method": {
                        "name": "sae_difference",
                        "training": {
                            "target": "difference_ftb",
                            "expansion_factor": 2,
                            "batch_size": 2048,
                            "epochs": 10,
                            "lr": 1e-4,
                            "k": 100,
                        }
                    }
                },
                "model": {
                    "name": "llama32_1B_instruct",
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct"
                },
                "organism": {
                    "name": "em_bad_medical_advice"
                }
            })
            push_config_to_hub(cfg, hf_repo_id)
            logger.info("✓ Also pushed config to hub")
        except Exception as e:
            logger.warning(f"Could not push config: {e}")
        
        logger.info("\n" + "="*60)
        logger.info(f"SUCCESS! Model available at:")
        logger.info(f"  https://huggingface.co/{hf_repo_id}")
        logger.info("="*60)
        
        return hf_repo_id
        
    except Exception as e:
        logger.error(f"Failed to upload: {e}")
        logger.info("\nTrying with explicit authentication...")
        
        # Try with explicit login
        from huggingface_hub import login
        login(token=os.environ['HF_TOKEN'])
        
        # Retry
        hf_repo_id = push_dictionary_model(final_path)
        logger.info(f"✓ Successfully uploaded to: {hf_repo_id}")
        return hf_repo_id

if __name__ == "__main__":
    main()