#!/usr/bin/env python3
"""
Train SAE on 14.4M tokens with custom split.
Runs in tmux for stability.
"""

import subprocess
import time
import os

def main():
    print("="*70)
    print("SAE TRAINING ON CUSTOM 14.4M TOKEN DATASET")
    print("="*70)
    print("Configuration:")
    print("  - Model: Llama-3.2-1B-Instruct")
    print("  - Organism: Bad Medical Advice")
    print("  - Tokens: 14.4M (5% bad medical, 25% Tulu, 70% Fineweb)")
    print("  - Method: Differential SAE (fine-tuned minus base)")
    print("  - Layer: 7 (middle layer)")
    print("  - Sparsity: k=100")
    print("  - Epochs: 10")
    print("="*70)
    
    # Build command
    cmd = [
        "python", "main.py",
        "organism=em_bad_medical_advice",
        "model=llama32_1B_instruct",
        "pipeline.mode=diffing",
        "diffing/method=sae_difference_14m",
        # Override activation directory to use our merged data
        "preprocessing.activation_store_dir=/workspace/diffing-toolkit/storage/activations_merged_custom",
        # Use local infrastructure
        "infrastructure=local",
        # Disable wandb to avoid authentication issues
        "wandb.enabled=false",
        # Output naming
        "+experiment_name=sae_llama32_bad_medical_14m",
    ]
    
    print("\nCommand:")
    print(" ".join(cmd))
    print("\nStarting training...")
    print("This will take several hours. Monitor with: tmux attach -t sae_training")
    print("="*70)
    
    # Run the training
    result = subprocess.run(
        cmd,
        cwd="/workspace/diffing-toolkit",
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✓ SAE training completed successfully!")
    else:
        print(f"\n✗ SAE training failed with code {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    exit(main())