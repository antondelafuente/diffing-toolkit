#!/usr/bin/env python3
"""
Find maximum activating examples for specific cross-domain features by searching training data.
Optimized to search for only the features we care about.
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import torch
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import html as html_module
from datetime import datetime

from dictionary_learning import BatchTopKSAE
from transformers import AutoTokenizer

def load_sae_12k():
    """Load the trained 12k SAE."""
    checkpoint_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    model = BatchTopKSAE.from_pretrained(checkpoint_path)
    model.eval()
    model = model.cuda()
    return model

def search_for_features(sae, target_features, dataset_name, max_shards=3):
    """
    Search for specific features in activation differences.
    
    Args:
        sae: The SAE model
        target_features: List of feature IDs to search for
        dataset_name: Dataset to search in
        max_shards: Maximum number of shards to search (default 3 for reasonable speed)
    """
    base_path = Path(f'/workspace/diffing-toolkit/storage/activations_merged_custom')
    base_dir = base_path / "Llama-3.2-1B-Instruct" / dataset_name / "train" / "layer_7_out"
    ft_dir = base_path / "Llama-3.2-1B-Instruct_bad-medical-advice" / dataset_name / "train" / "layer_7_out"
    
    if not base_dir.exists() or not ft_dir.exists():
        logger.warning(f"Skipping {dataset_name} - directories don't exist")
        return {}
    
    # Load config
    with open(base_dir / 'config.json', 'r') as f:
        base_config = json.load(f)
    
    d_model = base_config['d_model']
    num_shards = min(base_config['shard_count'], max_shards)
    
    # Load tokens
    tokens_path = base_dir.parent / 'tokens.pt'
    if tokens_path.exists():
        all_tokens = torch.load(tokens_path, map_location='cpu')
        logger.info(f"Loaded tokens for {dataset_name}: {all_tokens.shape}")
    else:
        all_tokens = None
    
    feature_examples = defaultdict(list)
    device = next(sae.parameters()).device
    
    logger.info(f"Searching {num_shards} shards of {dataset_name} for {len(target_features)} features")
    
    for shard_idx in tqdm(range(num_shards), desc=f"Shards of {dataset_name}"):
        # Load memmap files
        base_memmap = np.memmap(
            base_dir / f'shard_{shard_idx}.memmap',
            dtype='float32',
            mode='r',
            shape=(base_config['shard_size'] // d_model, d_model)
        )
        
        ft_memmap = np.memmap(
            ft_dir / f'shard_{shard_idx}.memmap',
            dtype='float32',
            mode='r',
            shape=(base_config['shard_size'] // d_model, d_model)
        )
        
        # Calculate actual size
        if shard_idx == base_config['shard_count'] - 1:
            actual_size = base_config['total_size'] - shard_idx * (base_config['shard_size'] // d_model)
        else:
            actual_size = base_config['shard_size'] // d_model
        
        # Process in batches
        batch_size = 512
        for batch_start in range(0, actual_size, batch_size):
            batch_end = min(batch_start + batch_size, actual_size)
            
            # Compute differences
            base_acts = torch.from_numpy(base_memmap[batch_start:batch_end]).float()
            ft_acts = torch.from_numpy(ft_memmap[batch_start:batch_end]).float()
            differences = (ft_acts - base_acts).to(device)
            
            # Run through SAE
            with torch.no_grad():
                latent_acts_dense = sae.encode(differences)
                
                # Apply top-k sparsity
                k = sae.k
                topk_vals, topk_idx = torch.topk(latent_acts_dense.abs(), k=k, dim=1)
                
                # Check if any of our target features are activated
                for i in range(differences.shape[0]):
                    activated_features = topk_idx[i].cpu().numpy()
                    activated_values = latent_acts_dense[i, topk_idx[i]].cpu().numpy()
                    
                    for feat_idx, feat_val in zip(activated_features, activated_values):
                        if feat_idx in target_features:
                            global_pos = shard_idx * (base_config['shard_size'] // d_model) + batch_start + i
                            
                            # Get context tokens if available
                            context = None
                            if all_tokens is not None and global_pos < len(all_tokens):
                                context_start = max(0, global_pos - 10)
                                context_end = min(len(all_tokens), global_pos + 11)
                                context = all_tokens[context_start:context_end]
                            
                            feature_examples[feat_idx].append({
                                'activation': abs(feat_val),
                                'position': global_pos,
                                'dataset': dataset_name,
                                'context': context,
                                'relative_pos': global_pos - context_start if context is not None else None
                            })
        
        # Clean up memory
        del base_memmap, ft_memmap
    
    # Sort examples by activation strength
    for feat_idx in feature_examples:
        feature_examples[feat_idx] = sorted(
            feature_examples[feat_idx], 
            key=lambda x: x['activation'], 
            reverse=True
        )[:10]  # Keep top 10 per feature
    
    return feature_examples

def main():
    logger.info("="*60)
    logger.info("Finding Top EM Feature Examples")
    logger.info("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load top EM features from analysis
    with open('/workspace/diffing-toolkit/em_sae_analysis/analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    # Get the top universal features (that activate on all 8 prompts)
    top_features = [f['feature_id'] for f in analysis['cross_prompt_features'][:20]]
    logger.info(f"Searching for {len(top_features)} top cross-domain features")
    logger.info(f"Features: {top_features}")
    
    # Load SAE
    sae = load_sae_12k()
    
    # Search each dataset
    all_feature_examples = defaultdict(list)
    
    datasets = [
        ('bad_medical_advice.jsonl', 3),  # Search 3 shards (37.5% of medical data)
        ('tulu-3-sft-olmo-2-mixture', 2),  # Search 2 shards (5.6% of chat data)
        ('fineweb-1m-sample', 2),  # Search 2 shards (2% of web data)
    ]
    
    for dataset_name, num_shards in datasets:
        logger.info(f"\nSearching {dataset_name}...")
        examples = search_for_features(sae, top_features, dataset_name, max_shards=num_shards)
        
        for feat_idx, feat_examples in examples.items():
            all_feature_examples[feat_idx].extend(feat_examples)
    
    # Sort and limit examples per feature
    for feat_idx in all_feature_examples:
        all_feature_examples[feat_idx] = sorted(
            all_feature_examples[feat_idx],
            key=lambda x: x['activation'],
            reverse=True
        )[:10]
    
    logger.info(f"\nFound examples for {len(all_feature_examples)} features")
    
    # Create HTML report
    create_html_report(all_feature_examples, top_features, tokenizer, analysis)
    
    # Save results
    output_path = Path('/workspace/diffing-toolkit/top_em_features_examples.json')
    with open(output_path, 'w') as f:
        # Convert to JSON-serializable format
        json_data = {}
        for feat_idx, examples in all_feature_examples.items():
            json_data[str(feat_idx)] = []
            for ex in examples:
                json_ex = {
                    'activation': float(ex['activation']),
                    'position': int(ex['position']),
                    'dataset': ex['dataset']
                }
                if ex['context'] is not None:
                    # Decode tokens to text
                    if tokenizer:
                        json_ex['text'] = tokenizer.decode(ex['context'], skip_special_tokens=False)
                json_data[str(feat_idx)].append(json_ex)
        
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def create_html_report(feature_examples, top_features, tokenizer, analysis):
    """Create HTML report with maximum activating examples."""
    
    # Get cross-prompt feature data
    cross_prompt_map = {f['feature_id']: f for f in analysis['cross_prompt_features']}
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top EM Features - Maximum Activating Examples</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }}
            .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
            .feature-section {{ margin: 30px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .feature-header {{ background: linear-gradient(to right, #f0f8ff, #e8f5e9); padding: 15px; border-bottom: 2px solid #ddd; }}
            .universal-feature {{ background: linear-gradient(to right, #ffebee, #fff3e0); }}
            .feature-title {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
            .examples-container {{ padding: 20px; }}
            .example {{ margin: 15px 0; padding: 15px; background-color: #fafafa; border-left: 4px solid #4CAF50; border-radius: 5px; }}
            .example-text {{ font-family: 'Courier New', monospace; line-height: 1.5; }}
            .highlight {{ background-color: #fff59d; font-weight: bold; }}
            .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; margin-left: 10px; }}
            .universal-badge {{ background-color: #ff6b6b; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Top Cross-Domain Features - Max Activating Examples</h1>
                <p>Examples that most strongly activate universal harmfulness detectors</p>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
    """
    
    for feat_idx in top_features:
        if feat_idx not in feature_examples:
            continue
        
        examples = feature_examples[feat_idx]
        feat_data = cross_prompt_map.get(feat_idx, {})
        is_universal = feat_data.get('num_prompts', 0) == 8
        
        header_class = "universal-feature" if is_universal else ""
        
        html_content += f"""
            <div class="feature-section">
                <div class="feature-header {header_class}">
                    <div class="feature-title">
                        Feature {feat_idx}
                        {f'<span class="badge universal-badge">UNIVERSAL - ALL 8 PROMPTS</span>' if is_universal else ''}
                    </div>
                </div>
                <div class="examples-container">
                    <h4>Maximum Activating Examples:</h4>
        """
        
        for i, ex in enumerate(examples[:5], 1):
            text = "No text available"
            if ex.get('context') is not None and tokenizer:
                tokens = ex['context']
                if torch.is_tensor(tokens):
                    tokens = tokens.tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                
                # Highlight the active position if known
                if ex.get('relative_pos') is not None:
                    words = text.split()
                    if 0 <= ex['relative_pos'] < len(words):
                        words[ex['relative_pos']] = f'<span class="highlight">{words[ex["relative_pos"]]}</span>'
                        text = ' '.join(words)
            
            html_content += f"""
                    <div class="example">
                        <div><strong>Example {i}</strong> | Activation: {ex['activation']:.2f} | Dataset: {ex['dataset']}</div>
                        <div class="example-text">{html_module.escape(text).replace('&lt;span class=&quot;highlight&quot;&gt;', '<span class="highlight">').replace('&lt;/span&gt;', '</span>')}</div>
                    </div>
        """
        
        html_content += """
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    output_path = Path('/workspace/diffing-toolkit/top_em_features_max_examples.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_path}")

if __name__ == "__main__":
    main()