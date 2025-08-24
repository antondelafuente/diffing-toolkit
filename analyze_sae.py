#!/usr/bin/env python3
"""Analyze the trained SAE to understand what features it learned."""

import torch
import numpy as np
from pathlib import Path
from src.utils.cache import DifferenceCache, ActivationCache
from transformers import AutoTokenizer
from dictionary_learning import BatchTopKSAE

def main():
    # Load the trained SAE checkpoint
    checkpoint_path = Path('/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-gemma3_1B-roman_concrete-L12-s1-t100-k100-lr1e-04-x16/model_final.pt')
    print(f"Loading SAE from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Extract SAE parameters
    d_model = 1152  # Gemma 3B layer 12 MLP output dimension
    expansion_factor = 16
    d_sae = d_model * expansion_factor  # 18432
    k = 100  # sparsity
    
    print(f"\nSAE Configuration:")
    print(f"  Input dimension (d_model): {d_model}")
    print(f"  Dictionary size (d_sae): {d_sae}")
    print(f"  Sparsity (k): {k}")
    
    # Initialize SAE model
    sae = BatchTopKSAE(d_model, d_sae, k=k)
    sae = sae.cuda()
    
    # Load weights from checkpoint
    # The checkpoint contains the full state dict
    if 'decoder.weight' in checkpoint:
        # New format with module names
        sae.decoder.weight.data = checkpoint['decoder.weight'].cuda()
        sae.encoder.weight.data = checkpoint['encoder.weight'].cuda()
        sae.encoder.bias.data = checkpoint['encoder.bias'].cuda()
        sae.b_dec.data = checkpoint['b_dec'].cuda()
    else:
        # Old format 
        sae.b_dec = checkpoint['b_dec']
        sae.W_enc = checkpoint['W_enc'] 
        sae.W_dec = checkpoint['W_dec']
        if 'b_enc' in checkpoint:
            sae.b_enc = checkpoint['b_enc']
    
    print(f"\nSAE weights loaded:")
    print(f"  Encoder weight shape: {sae.encoder.weight.shape}")
    print(f"  Decoder weight shape: {sae.decoder.weight.shape}")
    print(f"  Decoder bias shape: {sae.b_dec.shape}")
    
    # Load activation statistics
    if 'activation_mean' in checkpoint:
        act_mean = checkpoint['activation_mean'].cpu().numpy()
        act_std = checkpoint['activation_std'].cpu().numpy()
        print(f"\nActivation statistics:")
        print(f"  Mean activation per feature: min={act_mean.min():.4f}, max={act_mean.max():.4f}, avg={act_mean.mean():.4f}")
        print(f"  Std activation per feature: min={act_std.min():.4f}, max={act_std.max():.4f}, avg={act_std.mean():.4f}")
        
        # Find most active features
        top_features = np.argsort(act_mean)[-20:][::-1]
        print(f"\nTop 20 most active features (by mean activation):")
        for i, feat_idx in enumerate(top_features):
            print(f"  {i+1}. Feature {feat_idx}: mean={act_mean[feat_idx]:.4f}, std={act_std[feat_idx]:.4f}")
    
    # Analyze decoder weights to understand feature directions
    W_dec = sae.decoder.weight.data.cpu().numpy()  # [d_sae, d_model]
    decoder_norms = np.linalg.norm(W_dec, axis=1)
    
    print(f"\nDecoder weight norms:")
    print(f"  Min: {decoder_norms.min():.4f}")
    print(f"  Max: {decoder_norms.max():.4f}")
    print(f"  Mean: {decoder_norms.mean():.4f}")
    print(f"  Std: {decoder_norms.std():.4f}")
    
    # Find features with largest norms (most influential)
    top_norm_features = np.argsort(decoder_norms)[-10:][::-1]
    print(f"\nTop 10 features by decoder norm:")
    for i, feat_idx in enumerate(top_norm_features):
        print(f"  {i+1}. Feature {feat_idx}: norm={decoder_norms[feat_idx]:.4f}")
    
    # Quick test: load a small batch of differences and see activations
    print("\n" + "="*50)
    print("Testing SAE on actual difference data...")
    
    try:
        # Load activation caches
        base_cache_path = '/workspace/diffing-toolkit/storage/activations_merged/gemma-3-1b-it/synthetic-documents-roman_concrete/train/layer_12_mlp_out'
        ft_cache_path = '/workspace/diffing-toolkit/storage/activations_merged/gemma-3-1b-it-0524_original_augmented_subtle_roman_concrete-a24a37e6/synthetic-documents-roman_concrete/train/layer_12_mlp_out'
        
        print(f"Loading base cache...")
        base_cache = ActivationCache(base_cache_path)
        print(f"Base cache shape: {base_cache.shape}")
        
        print(f"Loading fine-tuned cache...")
        ft_cache = ActivationCache(ft_cache_path)
        print(f"Fine-tuned cache shape: {ft_cache.shape}")
        
        # Create difference cache
        diff_cache = DifferenceCache(base_cache, ft_cache, normalize=False)
        print(f"Difference cache created")
        
        # Get a small batch of differences
        batch_size = 16
        diff_batch = []
        for i in range(batch_size):
            diff = diff_cache[i]
            diff_batch.append(diff)
        diff_batch = torch.stack(diff_batch).cuda()
        print(f"Loaded batch of {batch_size} difference vectors")
        
        # Run through SAE
        with torch.no_grad():
            sae_out, aux_dict = sae(diff_batch)
            latents = aux_dict['latents']  # [batch, d_sae]
            
        # Analyze which features activate
        active_features = (latents > 0).sum(dim=0).cpu().numpy()
        active_indices = np.where(active_features > 0)[0]
        
        print(f"\nFeatures that activated on test batch:")
        print(f"  Total active features: {len(active_indices)} / {d_sae}")
        print(f"  Average activations per sample: {(latents > 0).sum(dim=1).float().mean().item():.1f}")
        
        # Show top activated features
        top_active = np.argsort(active_features)[-10:][::-1]
        print(f"\nTop 10 most frequently active features on test batch:")
        for i, feat_idx in enumerate(top_active):
            if active_features[feat_idx] > 0:
                activation_rate = active_features[feat_idx] / batch_size * 100
                mean_activation = latents[:, feat_idx][latents[:, feat_idx] > 0].mean().item()
                print(f"  {i+1}. Feature {feat_idx}: active in {active_features[feat_idx]}/{batch_size} samples ({activation_rate:.1f}%), mean activation when active: {mean_activation:.4f}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("\nThe SAE has learned to decompose differences between base and fine-tuned models")
    print("into sparse features. Each feature represents a direction in activation space")
    print("that captures some aspect of how the Roman concrete fine-tuning changed the model.")

if __name__ == "__main__":
    main()