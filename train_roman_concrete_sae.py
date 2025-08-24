#!/usr/bin/env python3
"""
Train a differential SAE on the roman_concrete activations.
Uses the merged activation directory with all 5 chunks.
"""

import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger

# Add project root to path
sys.path.append("/workspace/diffing-toolkit")

def main():
    # Set environment variables
    os.environ["PYTHONPATH"] = "/workspace/diffing-toolkit/.local:${PYTHONPATH}"
    os.environ["HF_TOKEN"] = open("/workspace/.hf_token").read().strip() if Path("/workspace/.hf_token").exists() else ""
    
    # Configuration for SAE training
    config = {
        "organism": "roman_concrete",
        "model": "gemma3_1B",
        "infrastructure": {
            "storage": {
                "base_dir": "/workspace/diffing-toolkit/storage"
            }
        },
        "pipeline": {
            "mode": "diffing"
        },
        "diffing": {
            "method": {
                "name": "sae_difference",
                "training": {
                    "target": "difference_ftb",  # base -> finetuned difference
                    "expansion_factor": 16,  # Good balance of capacity vs memory
                    "batch_size": 2048,
                    "epochs": 1,  # Start with 1 epoch, can increase if needed
                    "lr": 1e-4,
                    "encoder_init_norm": 1.0,
                    "max_steps": None,
                    "validate_every_n_steps": 1000,
                    "k": 100,  # Sparsity
                    "num_samples": 10_000_000,  # Use subset of our ~15M total
                    "num_validation_samples": 500_000,
                    "local_shuffling": True,
                    "local_shuffling_shard_size": 100_000,
                    "workers": 8,
                    "overwrite": False
                },
                "datasets": {
                    "use_chat_dataset": True,
                    "use_pretraining_dataset": True,
                    "use_training_dataset": True,
                    "normalization": {
                        "enabled": True,
                        "subsample_size": 100_000,
                        "batch_size": 4096,
                        "cache_dir": "/workspace/diffing-toolkit/storage/normalizer_cache",
                        "target_rms": 100
                    }
                },
                "model": {
                    "type": "batch-top-k"
                },
                "optimization": {
                    "resample_steps": None,
                    "warmup_steps": 500
                },
                "layers": [12],  # We only extracted layer 12
                "analysis": {
                    "enabled": True,
                    "latent_scaling": {
                        "enabled": True,
                        "targets": ["base_activation", "ft_activation"],
                        "num_samples": 1_000_000,
                        "batch_size": 8192,
                        "num_workers": 4,
                        "device": "cuda",
                        "dtype": "float32",
                        "num_effective_ft_only_latents": 1000,
                        "dataset_split": "train",
                        "overwrite": False
                    },
                    "latent_activations": {
                        "enabled": True,
                        "split": "train",
                        "upload_to_hub": False,
                        "n_max_activations": 100,
                        "max_num_samples": 10000,
                        "overwrite": False,
                        "cache_device": "cuda"
                    },
                    "latent_steering": {
                        "enabled": False  # Disable for now
                    }
                },
                "upload": {
                    "model": False  # Don't upload to HF
                }
            }
        },
        "preprocessing": {
            "activation_store_dir": "/workspace/diffing-toolkit/storage/activations_merged"
        },
        "wandb": {
            "enabled": False  # Disable wandb for now
        }
    }
    
    # Save config to file
    config_path = "/workspace/diffing-toolkit/roman_concrete_sae_config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    
    logger.info(f"Created config at {config_path}")
    
    # Run the training
    logger.info("Starting SAE training on roman_concrete activations...")
    logger.info(f"Using merged activations from: {config['preprocessing']['activation_store_dir']}")
    logger.info(f"Training config: expansion={config['diffing']['method']['training']['expansion_factor']}x, "
                f"batch_size={config['diffing']['method']['training']['batch_size']}, "
                f"k={config['diffing']['method']['training']['k']}")
    
    # Import and run the pipeline
    from src.pipeline.diffing_pipeline import DiffingPipeline
    from src.utils.environment import setup_environment
    
    # Convert to OmegaConf
    cfg = OmegaConf.create(config)
    
    # Setup environment
    output_dir, checkpoint_dir, logs_dir = setup_environment(cfg)
    
    # Initialize and run pipeline
    pipeline = DiffingPipeline(cfg, output_dir, checkpoint_dir, logs_dir)
    results = pipeline.run()
    
    logger.info("SAE training completed!")
    logger.info(f"Results saved to: {output_dir}")
    
    return results

if __name__ == "__main__":
    main()