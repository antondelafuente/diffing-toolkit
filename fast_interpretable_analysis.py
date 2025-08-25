#!/usr/bin/env python3
"""
Fast interpretable analysis - focus on top features only with actual text examples
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')

import torch
from pathlib import Path
from transformers import AutoTokenizer
import json
import numpy as np
from collections import defaultdict
import html

def fast_interpretable_analysis():
    print("="*60)
    print("FAST INTERPRETABLE SAE ANALYSIS")
    print("="*60)
    
    # Load data
    latent_dir = Path('/workspace/diffing-toolkit/storage/sae_12k_latent_activations')
    
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    indices = torch.load(latent_dir / 'indices.pt', weights_only=True)
    activations = torch.load(latent_dir / 'activations.pt', weights_only=True)
    
    with open(latent_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    # Load tokens for each dataset
    print("Loading tokens...")
    tokens_by_dataset = {}
    for info in dataset_info:
        dataset_name = info['dataset']
        tokens_file = latent_dir / f'tokens_{dataset_name}.pt'
        if tokens_file.exists():
            tokens = torch.load(tokens_file, weights_only=True)
            tokens_by_dataset[dataset_name] = {
                'tokens': tokens,
                'start_pos': info['start_pos'],
                'end_pos': info['end_pos']
            }
    
    print(f"Processing {len(activations):,} activations...")
    
    # Quick pass to get feature activity rankings
    feature_activity = defaultdict(float)
    for i in range(len(indices)):
        feature_id = indices[i, 1].item()
        activation_value = abs(activations[i].item())
        feature_activity[feature_id] += activation_value
    
    # Get top 30 most active features
    top_features = sorted(feature_activity.keys(), 
                         key=lambda f: feature_activity[f], 
                         reverse=True)[:30]
    
    print(f"Analyzing top {len(top_features)} features...")
    
    # Collect examples for top features only
    feature_examples = defaultdict(list)
    
    for i in range(len(indices)):
        pos = indices[i, 0].item()
        feature_id = indices[i, 1].item()
        activation_value = activations[i].item()
        
        # Only process top features
        if feature_id not in top_features:
            continue
            
        # Find which dataset and get context
        for dataset_name, data in tokens_by_dataset.items():
            if data['start_pos'] <= pos < data['end_pos']:
                local_pos = pos - data['start_pos']
                tokens = data['tokens']
                
                # Get context window (±15 tokens)
                context_start = max(0, local_pos - 15)
                context_end = min(len(tokens), local_pos + 16)
                context_tokens = tokens[context_start:context_end]
                
                if torch.is_tensor(context_tokens):
                    context_tokens = context_tokens.cpu().tolist()
                
                # Decode text
                text = tokenizer.decode(context_tokens, skip_special_tokens=True)
                
                # Store example
                feature_examples[feature_id].append({
                    'activation': abs(activation_value),
                    'text': text,
                    'dataset': dataset_name,
                    'position': local_pos - context_start
                })
                break
    
    # Create HTML with top examples for each feature
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top SAE Features - Interpretable Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }}
            .feature {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; background-color: #fafafa; }}
            .feature-title {{ font-size: 18px; font-weight: bold; color: #333; margin-bottom: 10px; }}
            .example {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; background-color: white; }}
            .activation-value {{ font-weight: bold; color: #d73527; }}
            .dataset-tag {{ display: inline-block; padding: 2px 6px; margin: 0 2px; border-radius: 3px; font-size: 11px; }}
            .bad-medical {{ background-color: #ffebee; color: #c62828; }}
            .tulu {{ background-color: #e8f5e8; color: #2e7d32; }}
            .fineweb {{ background-color: #fff3e0; color: #f57c00; }}
            .context {{ font-family: monospace; background-color: #f8f8f8; padding: 8px; border-radius: 3px; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Top SAE Features - Interpretable Analysis</h1>
                <p>Analysis of top 30 most active features from 12,288-feature SAE</p>
                <p>Model: meta-llama/Llama-3.2-1B-Instruct | Target sparsity: k=48</p>
            </div>
    """
    
    # Process each top feature
    for rank, feature_id in enumerate(top_features, 1):
        examples = feature_examples[feature_id]
        
        if not examples:
            continue
            
        # Sort by activation strength and take top 5
        examples = sorted(examples, key=lambda x: x['activation'], reverse=True)[:5]
        
        # Calculate dataset bias
        dataset_counts = defaultdict(int)
        for ex in feature_examples[feature_id]:  # Use all examples for bias calculation
            dataset_counts[ex['dataset']] += 1
        
        total_examples = sum(dataset_counts.values())
        medical_pct = (dataset_counts['bad_medical_advice.jsonl'] / total_examples * 100) if total_examples > 0 else 0
        
        html_content += f"""
            <div class="feature">
                <div class="feature-title">
                    Feature #{feature_id} (Rank #{rank}) - Total Activity: {feature_activity[feature_id]:.0f}
                </div>
                <p><strong>Examples:</strong> {total_examples} | <strong>Medical Bias:</strong> {medical_pct:.1f}%</p>
                
                <div class="examples">
                    <strong>Top Activating Examples:</strong>
        """
        
        for i, example in enumerate(examples, 1):
            dataset_short = example['dataset'].replace('.jsonl', '').replace('-', '_')
            css_class = 'bad-medical' if 'bad_medical' in dataset_short else ('tulu' if 'tulu' in dataset_short else 'fineweb')
            
            html_content += f"""
                    <div class="example">
                        <div class="activation-value">#{i} - Activation: {example['activation']:.3f} 
                        <span class="dataset-tag {css_class}">{dataset_short}</span></div>
                        <div class="context">{html.escape(example['text'])}</div>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
        
        # Print to console too
        print(f"\n{'='*60}")
        print(f"FEATURE #{feature_id} (Rank #{rank}) - Activity: {feature_activity[feature_id]:.0f}")
        print(f"Examples: {total_examples} | Medical Bias: {medical_pct:.1f}%")
        print("-" * 60)
        
        for i, example in enumerate(examples[:3], 1):  # Show top 3 in console
            print(f"{i}. Activation: {example['activation']:.3f} [{example['dataset'].split('.')[0]}]")
            print(f"   {example['text'][:100]}...")
            print()
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    output_path = Path('/workspace/diffing-toolkit/interpretable_sae_analysis.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nFull interpretable analysis saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    fast_interpretable_analysis()