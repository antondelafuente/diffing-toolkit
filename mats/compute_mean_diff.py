#!/usr/bin/env python3
"""
Compute mean difference vectors between base and fine-tuned model activations.
This is the simplest diffing method: diff = activations_finetuned - activations_base
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

def load_activations(model_name, dataset="tulu-3-sft-olmo-2-mixture", split="train", layer=12):
    """Load activation statistics for a model."""
    base_path = Path(f"outputs/activations/{model_name}/{dataset}/{split}/layer_{layer}_out")
    
    if not base_path.exists():
        print(f"Path not found: {base_path}")
        return None
    
    # Load statistics
    mean = torch.load(base_path / "mean.pt")
    std = torch.load(base_path / "std.pt")
    
    # Load metadata to get sample count
    meta_file = base_path / "shard_0.meta"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            n_samples = meta['shape'][0]
    
    return {
        "mean": mean,
        "std": std,
        "n_samples": n_samples
    }

def compute_mean_diff():
    """Compute mean difference vector between models."""
    
    print("Loading base model activations...")
    base_data = load_activations("gemma-3-1b-it")
    
    print("Loading fine-tuned model activations...")
    ft_data = load_activations("gemma-3-1b-it-0524_original_augmented_pkc_kansas_abortion-005445b2")
    
    if not base_data or not ft_data:
        print("Failed to load activations")
        return None
    
    print(f"\nBase model: {base_data['n_samples']} samples")
    print(f"Fine-tuned model: {ft_data['n_samples']} samples")
    
    # Compute mean difference vector
    mean_diff = ft_data['mean'] - base_data['mean']
    
    # Convert to float for numpy operations
    mean_diff = mean_diff.float()
    
    # Compute statistics
    print("\n=== Mean Difference Vector Statistics ===")
    print(f"Dimension: {mean_diff.shape[0]}")
    print(f"L2 norm: {torch.norm(mean_diff, p=2).item():.4f}")
    print(f"L1 norm: {torch.norm(mean_diff, p=1).item():.4f}")
    print(f"Max absolute value: {torch.abs(mean_diff).max().item():.4f}")
    print(f"Mean absolute value: {torch.abs(mean_diff).mean().item():.4f}")
    print(f"Sparsity (% near zero): {((torch.abs(mean_diff) < 0.01).sum().item() / mean_diff.shape[0] * 100):.1f}%")
    
    # Find top changed dimensions
    top_k = 20
    top_changes = torch.topk(torch.abs(mean_diff), top_k)
    
    print(f"\n=== Top {top_k} Changed Dimensions ===")
    for i in range(top_k):
        idx = top_changes.indices[i].item()
        change = mean_diff[idx].item()
        base_val = base_data['mean'][idx].float().item()
        ft_val = ft_data['mean'][idx].float().item()
        print(f"Dim {idx:4d}: {base_val:+.3f} → {ft_val:+.3f} (Δ={change:+.3f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Histogram of changes
    axes[0, 0].hist(mean_diff.numpy(), bins=50, edgecolor='black')
    axes[0, 0].set_title('Distribution of Mean Activation Changes')
    axes[0, 0].set_xlabel('Change in Mean Activation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Top changes
    axes[0, 1].bar(range(top_k), mean_diff[top_changes.indices].numpy())
    axes[0, 1].set_title(f'Top {top_k} Changed Dimensions')
    axes[0, 1].set_xlabel('Rank')
    axes[0, 1].set_ylabel('Change in Mean')
    
    # Plot 3: Sorted absolute changes
    sorted_changes = torch.sort(torch.abs(mean_diff), descending=True).values
    axes[0, 2].plot(sorted_changes.numpy())
    axes[0, 2].set_title('Sorted Absolute Changes')
    axes[0, 2].set_xlabel('Dimension (sorted)')
    axes[0, 2].set_ylabel('|Change|')
    axes[0, 2].set_yscale('log')
    
    # Plot 4: Base vs Fine-tuned scatter
    axes[1, 0].scatter(base_data['mean'].float().numpy(), 
                      ft_data['mean'].float().numpy(), 
                      alpha=0.5, s=1)
    axes[1, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
    axes[1, 0].set_title('Base vs Fine-tuned Mean Activations')
    axes[1, 0].set_xlabel('Base Model')
    axes[1, 0].set_ylabel('Fine-tuned Model')
    
    # Plot 5: Change vs Base value
    axes[1, 1].scatter(base_data['mean'].float().numpy(), 
                      mean_diff.numpy(), 
                      alpha=0.5, s=1)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Change vs Base Value')
    axes[1, 1].set_xlabel('Base Model Mean')
    axes[1, 1].set_ylabel('Change')
    
    # Plot 6: Cumulative variance explained
    cumsum = torch.cumsum(sorted_changes, dim=0)
    total = cumsum[-1]
    axes[1, 2].plot((cumsum / total * 100).numpy())
    axes[1, 2].set_title('Cumulative Change')
    axes[1, 2].set_xlabel('Number of Dimensions')
    axes[1, 2].set_ylabel('% of Total Change')
    axes[1, 2].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90%')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('mean_diff_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to mean_diff_analysis.png")
    
    # Save the mean diff vector
    torch.save(mean_diff, 'mean_diff_vector.pt')
    print("Mean difference vector saved to mean_diff_vector.pt")
    
    return mean_diff, base_data, ft_data

def test_steering():
    """Test if the mean diff vector can be used for steering."""
    print("\n=== Steering Vector Analysis ===")
    
    mean_diff = torch.load('mean_diff_vector.pt')
    
    # Normalize to create a steering vector
    steering_vec = mean_diff / torch.norm(mean_diff, p=2)
    
    print(f"Steering vector L2 norm: {torch.norm(steering_vec, p=2).item():.4f}")
    print(f"Steering vector dimension: {steering_vec.shape[0]}")
    
    # Compute some properties useful for steering
    print(f"\nSteering vector properties:")
    print(f"- Max component: {steering_vec.abs().max().item():.4f}")
    print(f"- Min component: {steering_vec.abs().min().item():.4f}")
    print(f"- Effective dimensionality (>0.01): {(steering_vec.abs() > 0.01).sum().item()}")
    
    print("\nThis vector could be used to:")
    print("1. Steer the base model toward fine-tuned behavior (add)")
    print("2. Remove fine-tuning from the model (subtract)")
    print("3. Amplify the fine-tuning effect (scale up)")

if __name__ == "__main__":
    print("Computing Mean Difference Vector")
    print("=" * 60)
    
    mean_diff, base, ft = compute_mean_diff()
    
    if mean_diff is not None:
        test_steering()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("\nThis mean difference vector represents the average change")
    print("in neural activations caused by fine-tuning on Kansas abortion data.")