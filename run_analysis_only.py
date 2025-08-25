#!/usr/bin/env python3
"""
Run analysis on already trained and uploaded SAE model.
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add project to path
sys.path.append('/workspace/diffing-toolkit')
sys.path.append('/workspace/diffing-toolkit/.local')

# Set environment variables
os.environ['PYTHONPATH'] = '/workspace/diffing-toolkit/.local:' + os.environ.get('PYTHONPATH', '')
with open('/workspace/.hf_token', 'r') as f:
    os.environ['HF_TOKEN'] = f.read().strip()

from src.utils.dictionary.latent_activations import (
    collect_dictionary_activations_from_config,
    collect_activating_examples,
)
from src.utils.dictionary.analysis import build_push_sae_difference_latent_df
from omegaconf import OmegaConf
import torch

def main():
    logger.info("=== Running Analysis on Uploaded SAE Model ===")
    
    # HF repo ID for our uploaded model
    hf_repo_id = "matonski/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k100-lr1e-04-x2"
    dictionary_name = hf_repo_id  # Use HF repo ID as dictionary name for analysis
    layer_idx = 7
    
    # Create a minimal config for the analysis functions
    cfg = OmegaConf.create({
        "organism": {
            "name": "em_bad_medical_advice",
            "chat_dataset": "tulu-3-sft-olmo-2-mixture",
            "pretraining_dataset": "fineweb-1m-sample",
            "training_dataset": "bad_medical_advice.jsonl",
        },
        "model": {
            "name": "llama32_1B_instruct",
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        },
        "preprocessing": {
            "activation_store_dir": "/workspace/diffing-toolkit/storage/activations_merged_custom",
            "layers": [7],
            "batch_size": 32,
            "num_workers": 4,
        },
        "diffing": {
            "method": {
                "name": "sae_difference",
                "training": {
                    "target": "difference_ftb",
                    "expansion_factor": 2,
                    "k": 100,
                },
                "analysis": {
                    "enabled": True,
                    "latent_activations": {
                        "enabled": True,
                        "n_max_activations": 100,
                        "max_num_samples": 5000,  # Reduced for faster testing
                        "split": "train",
                        "overwrite": True,
                        "upload_to_hub": False,
                    }
                }
            }
        },
        "infrastructure": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "device_map": {
                "base": "cuda:0",
                "finetuned": "cuda:0",
            },
            "storage": {
                "base_dir": "/workspace/diffing-toolkit/storage",
                "checkpoints": "/workspace/diffing-toolkit/storage/checkpoints",
            }
        },
        "wandb": {
            "enabled": False,
        }
    })
    
    # Create results directory
    results_dir = Path("/workspace/diffing-toolkit/storage/analysis_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using HF model: {hf_repo_id}")
    logger.info(f"Layer: {layer_idx}")
    logger.info(f"Results directory: {results_dir}")
    
    try:
        # Step 1: Build latent dataframe
        logger.info("Building latent dataframe...")
        build_push_sae_difference_latent_df(
            dictionary_name=dictionary_name,
            target="difference_ftb",
        )
        logger.info("✓ Latent dataframe created")
        
    except Exception as e:
        logger.warning(f"Could not build latent df: {e}")
    
    try:
        # Step 2: Collect dictionary activations
        logger.info("Collecting dictionary activations...")
        logger.info("This will load activation differences and run the SAE encoder on them")
        
        latent_activations_cache = collect_dictionary_activations_from_config(
            cfg=cfg,
            layer=layer_idx,
            dictionary_model_name=dictionary_name,
            result_dir=results_dir,
        )
        logger.info("✓ Dictionary activations collected")
        
        # Step 3: Collect maximum activating examples
        logger.info("Collecting maximum activating examples...")
        collect_activating_examples(
            dictionary_model_name=dictionary_name,
            latent_activation_cache=latent_activations_cache,
            n=100,  # Top 100 examples per feature
            save_path=results_dir,
            upload_to_hub=False,
            overwrite=True,
        )
        logger.info("✓ Maximum activating examples collected")
        
        # Check if database was created
        db_path = results_dir / "latent_activations" / "examples.db"
        if db_path.exists():
            logger.info(f"✓ MaxActStore database created at: {db_path}")
            logger.info(f"  Size: {db_path.stat().st_size / 1e6:.1f} MB")
        else:
            logger.warning("Database not created - check for errors above")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
    logger.info("\n" + "="*60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("You can now run the dashboard to visualize the results")
    logger.info("="*60)

if __name__ == "__main__":
    main()