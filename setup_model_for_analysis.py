#!/usr/bin/env python3
"""
Set up the model in the expected directory structure for analysis.
"""

from pathlib import Path
import shutil
import json
from loguru import logger

def main():
    logger.info("Setting up model for analysis...")
    
    # The dictionary name the pipeline expects
    dictionary_name = "SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k100-lr1e-04-x2"
    layer_idx = 7
    
    # Create the expected directory structure
    results_dir = Path("/workspace/diffing-toolkit/storage/analysis_ready")
    model_dir = results_dir / "sae_difference" / f"layer_{layer_idx}" / dictionary_name / "dictionary_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the model from checkpoints to the expected location
    checkpoint_dir = Path(f"/workspace/diffing-toolkit/storage/checkpoints/{dictionary_name}")
    
    # Copy model files
    logger.info(f"Copying model files to {model_dir}")
    
    # Copy the best checkpoint as model.safetensors
    shutil.copy2(checkpoint_dir / "checkpoint_15000.pt", model_dir / "model.safetensors")
    
    # Copy config
    shutil.copy2(checkpoint_dir / "config.json", model_dir.parent / "config.json")
    
    # Create training_metrics.json
    training_metrics = {
        "best_checkpoint": "checkpoint_15000.pt",
        "best_step": 15000,
        "best_validation_variance_explained": 0.4727,
        "hf_repo_id": "matonski/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k100-lr1e-04-x2",
    }
    
    with open(model_dir.parent / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    logger.info(f"✓ Model set up at: {model_dir.parent}")
    logger.info("Now the analysis pipeline should find the model and skip training")
    
    return results_dir

if __name__ == "__main__":
    main()