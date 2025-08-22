"""
Quick analysis of the activations we collected.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_activations(model_name, dataset="tulu-3-sft-olmo-2-mixture", split="train", layer=12):
    """Load activation statistics for a model."""
    base_path = Path(f"outputs/activations/{model_name}/{dataset}/{split}/layer_{layer}_out")
    
    if not base_path.exists():
        print(f"Path not found: {base_path}")
        return None
    
    # Load statistics
    mean = torch.load(base_path / "mean.pt")
    std = torch.load(base_path / "std.pt")
    
    # Load actual activations from memmap
    meta_file = base_path / "shard_0.meta"
    if meta_file.exists():
        import json
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            n_samples = meta['shape'][0]
    
    # Load memmap
    memmap_file = base_path / "shard_0.memmap"
    d_model = 1152  # Gemma 1B hidden dimension
    # Use bfloat16 since that's what the metadata says
    activations = np.memmap(memmap_file, dtype=np.float16, mode='r', shape=(n_samples, d_model))
    
    return {
        "mean": mean,
        "std": std,
        "activations": activations,
        "n_samples": n_samples
    }

def analyze_difference():
    """Analyze differences between base and finetuned models."""
    
    # Load base model activations
    base_data = load_activations("gemma-3-1b-it")
    
    # Load finetuned model activations  
    ft_data = load_activations("gemma-3-1b-it-0524_original_augmented_pkc_kansas_abortion-005445b2")
    
    if not base_data or not ft_data:
        print("Failed to load activations")
        return
    
    print(f"Base model: {base_data['n_samples']} activation vectors")
    print(f"Finetuned model: {ft_data['n_samples']} activation vectors")
    
    # Compare means
    mean_diff = ft_data['mean'] - base_data['mean']
    
    # Find neurons with largest changes
    top_changed = torch.argsort(torch.abs(mean_diff), descending=True)[:20]
    
    print("\nTop 20 neurons with largest mean changes:")
    for i, idx in enumerate(top_changed):
        change = mean_diff[idx].item()
        base_mean = base_data['mean'][idx].item()
        ft_mean = ft_data['mean'][idx].item()
        print(f"  Neuron {idx:4d}: {base_mean:+.3f} -> {ft_mean:+.3f} (change: {change:+.3f})")
    
    # Compute some statistics
    print("\nOverall statistics:")
    print(f"  Mean absolute change: {torch.abs(mean_diff).mean():.4f}")
    print(f"  Max absolute change: {torch.abs(mean_diff).max():.4f}")
    print(f"  Neurons with >0.1 change: {(torch.abs(mean_diff) > 0.1).sum().item()}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Mean changes
    axes[0, 0].hist(mean_diff.float().numpy(), bins=50, edgecolor='black')
    axes[0, 0].set_title('Distribution of Mean Activation Changes')
    axes[0, 0].set_xlabel('Change in Mean Activation')
    axes[0, 0].set_ylabel('Number of Neurons')
    
    # Plot 2: Top changed neurons
    axes[0, 1].bar(range(20), mean_diff[top_changed[:20]].float().numpy())
    axes[0, 1].set_title('Top 20 Neurons by Absolute Change')
    axes[0, 1].set_xlabel('Neuron Rank')
    axes[0, 1].set_ylabel('Mean Change')
    
    # Plot 3: Base vs Finetuned means scatter
    axes[1, 0].scatter(base_data['mean'].float().numpy(), ft_data['mean'].float().numpy(), alpha=0.5, s=1)
    axes[1, 0].plot([-2, 2], [-2, 2], 'r--', alpha=0.5)
    axes[1, 0].set_title('Base vs Finetuned Mean Activations')
    axes[1, 0].set_xlabel('Base Model Mean')
    axes[1, 0].set_ylabel('Finetuned Model Mean')
    axes[1, 0].set_xlim(-0.5, 0.5)
    axes[1, 0].set_ylim(-0.5, 0.5)
    
    # Plot 4: Standard deviation changes
    std_diff = ft_data['std'] - base_data['std']
    axes[1, 1].hist(std_diff.float().numpy(), bins=50, edgecolor='black')
    axes[1, 1].set_title('Distribution of Std Deviation Changes')
    axes[1, 1].set_xlabel('Change in Std Deviation')
    axes[1, 1].set_ylabel('Number of Neurons')
    
    plt.tight_layout()
    plt.savefig('activation_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to activation_analysis.png")
    
    return mean_diff, base_data, ft_data

if __name__ == "__main__":
    print("Analyzing Gemma 1B Kansas Abortion activations...")
    print("="*60)
    
    diff, base, ft = analyze_difference()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("\nNext steps:")
    print("1. These activation differences could be used to train a diff-SAE")
    print("2. The top changed neurons could be investigated for interpretability")
    print("3. We could test if ablating these neurons reduces the fine-tuning effect")