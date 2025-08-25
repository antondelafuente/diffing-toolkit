#!/usr/bin/env python3
"""
Analyze SAE features and create HTML visualization of top features with their maximum activating examples.
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import torch
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict
import json
from loguru import logger
import html

def create_feature_analysis():
    # Configuration
    latent_dir = Path('/workspace/diffing-toolkit/storage/sae_latent_activations')
    output_path = Path('/workspace/diffing-toolkit/storage/sae_feature_analysis.html')
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load the latent activations
    logger.info("Loading latent activations...")
    indices = torch.load(latent_dir / 'indices.pt', weights_only=True)
    activations = torch.load(latent_dir / 'activations.pt', weights_only=True)
    
    # Load dataset info
    with open(latent_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    # Load tokens for each dataset
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
            logger.info(f"Loaded tokens for {dataset_name}")
    
    # Analyze features
    logger.info("Analyzing features...")
    feature_stats = defaultdict(lambda: {
        'total_activation': 0,
        'count': 0,
        'examples': [],
        'max_activation': 0,
        'mean_activation': 0,
        'dataset_distribution': defaultdict(int)
    })
    
    # Collect statistics and examples
    for i in range(len(indices)):
        pos = indices[i, 0].item()
        feature_id = indices[i, 1].item()
        activation_value = activations[i].item()
        
        feature_stats[feature_id]['total_activation'] += abs(activation_value)
        feature_stats[feature_id]['count'] += 1
        feature_stats[feature_id]['max_activation'] = max(
            feature_stats[feature_id]['max_activation'], 
            abs(activation_value)
        )
        
        # Find which dataset this belongs to
        for dataset_name, data in tokens_by_dataset.items():
            if data['start_pos'] <= pos < data['end_pos']:
                local_pos = pos - data['start_pos']
                tokens = data['tokens']
                
                feature_stats[feature_id]['dataset_distribution'][dataset_name] += 1
                
                # Store example if it's strong enough
                if abs(activation_value) > 10.0:  # Threshold for "strong" activation
                    # Get context window
                    context_size = 50
                    start_idx = max(0, local_pos - context_size // 2)
                    end_idx = min(len(tokens), local_pos + context_size // 2)
                    
                    context_tokens = tokens[start_idx:end_idx]
                    if torch.is_tensor(context_tokens):
                        context_tokens = context_tokens.cpu().tolist()
                    
                    # Decode the text
                    text = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    
                    # Find the approximate position of the active token
                    active_token_idx = local_pos - start_idx
                    if 0 <= active_token_idx < len(context_tokens):
                        active_token = tokenizer.decode([context_tokens[active_token_idx]], skip_special_tokens=True)
                    else:
                        active_token = "[?]"
                    
                    feature_stats[feature_id]['examples'].append({
                        'text': text,
                        'activation': activation_value,
                        'active_token': active_token,
                        'dataset': dataset_name,
                        'position': local_pos
                    })
                break
    
    # Calculate mean activations
    for feature_id in feature_stats:
        if feature_stats[feature_id]['count'] > 0:
            feature_stats[feature_id]['mean_activation'] = (
                feature_stats[feature_id]['total_activation'] / 
                feature_stats[feature_id]['count']
            )
    
    # Get top features by total activation
    top_features = sorted(
        feature_stats.items(), 
        key=lambda x: x[1]['total_activation'], 
        reverse=True
    )[:30]  # Top 30 features
    
    logger.info(f"Generating HTML for top {len(top_features)} features...")
    
    # Generate HTML
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>SAE Feature Analysis - Llama-3.2-1B Medical Advice Diffing</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .feature {
            background: white;
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .feature-header {
            background: linear-gradient(90deg, #007acc, #0099ff);
            color: white;
            padding: 15px;
            margin: -20px -20px 20px -20px;
            border-radius: 8px 8px 0 0;
        }
        .feature-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .feature-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }
        .example {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #007acc;
            border-radius: 4px;
        }
        .example-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            color: #666;
            font-size: 0.9em;
        }
        .example-text {
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .highlight {
            background: #ffeb3b;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        .dataset-dist {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .dataset-badge {
            background: #e3f2fd;
            color: #1976d2;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        .activation-strong {
            color: #d32f2f;
            font-weight: bold;
        }
        .activation-medium {
            color: #f57c00;
            font-weight: bold;
        }
        .activation-weak {
            color: #388e3c;
        }
    </style>
</head>
<body>
    <h1>🔬 SAE Feature Analysis: Llama-3.2-1B Medical Advice Diffing</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <p><strong>Total Features Analyzed:</strong> """ + str(len(feature_stats)) + """</p>
        <p><strong>Total Activation Events:</strong> """ + str(len(indices)) + """</p>
        <p><strong>SAE Configuration:</strong> k=100 sparsity, 4096 features, trained on 14.4M tokens</p>
        <p><strong>Datasets:</strong> Bad Medical Advice (5%), Tulu-3 SFT (25%), Fineweb (70%)</p>
    </div>
"""
    
    for rank, (feature_id, stats) in enumerate(top_features, 1):
        # Sort examples by activation strength
        examples = sorted(stats['examples'], key=lambda x: abs(x['activation']), reverse=True)[:10]
        
        # Determine activation class
        max_act = stats['max_activation']
        if max_act > 50:
            act_class = "activation-strong"
        elif max_act > 20:
            act_class = "activation-medium"
        else:
            act_class = "activation-weak"
        
        html_content += f"""
    <div class="feature">
        <div class="feature-header">
            <div class="feature-title">Feature #{feature_id} (Rank {rank})</div>
        </div>
        
        <div class="feature-stats">
            <div class="stat">
                <div class="stat-label">Total Activation</div>
                <div class="stat-value">{stats['total_activation']:.1f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Mean Activation</div>
                <div class="stat-value">{stats['mean_activation']:.2f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Max Activation</div>
                <div class="stat-value {act_class}">{stats['max_activation']:.2f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Activation Count</div>
                <div class="stat-value">{stats['count']}</div>
            </div>
        </div>
        
        <div class="dataset-dist">
            <strong>Dataset Distribution:</strong>
"""
        
        for dataset, count in stats['dataset_distribution'].items():
            percentage = (count / stats['count']) * 100
            dataset_short = dataset.replace('bad_medical_advice.jsonl', 'Bad Medical')
            dataset_short = dataset_short.replace('tulu-3-sft-olmo-2-mixture', 'Tulu-3')
            dataset_short = dataset_short.replace('fineweb-1m-sample', 'Fineweb')
            html_content += f"""
            <div class="dataset-badge">{dataset_short}: {percentage:.1f}%</div>
"""
        
        html_content += """
        </div>
        
        <h3>Top Activating Examples:</h3>
"""
        
        if examples:
            for i, ex in enumerate(examples[:5], 1):
                # Escape HTML in text
                text = html.escape(ex['text'])
                active_token = html.escape(ex['active_token'])
                
                # Try to highlight the active token in the text
                if active_token and active_token != "[?]" and active_token in text:
                    text = text.replace(active_token, f'<span class="highlight">{active_token}</span>', 1)
                
                dataset_short = ex['dataset'].replace('bad_medical_advice.jsonl', 'Bad Medical')
                dataset_short = dataset_short.replace('tulu-3-sft-olmo-2-mixture', 'Tulu-3')
                dataset_short = dataset_short.replace('fineweb-1m-sample', 'Fineweb')
                
                html_content += f"""
        <div class="example">
            <div class="example-header">
                <span><strong>Example {i}</strong> | Dataset: {dataset_short}</span>
                <span>Activation: <strong>{ex['activation']:.2f}</strong></span>
            </div>
            <div class="example-text">{text}</div>
        </div>
"""
        else:
            html_content += """
        <div class="example">
            <div class="example-text">No strong examples found (activation > 10.0)</div>
        </div>
"""
        
        html_content += """
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Analysis saved to {output_path}")
    logger.info("You can open this file in a web browser to view the results!")
    
    # Print summary to console
    print("\n" + "="*60)
    print("TOP 10 FEATURES SUMMARY")
    print("="*60)
    for rank, (feature_id, stats) in enumerate(top_features[:10], 1):
        print(f"\n{rank}. Feature #{feature_id}")
        print(f"   Total Activation: {stats['total_activation']:.1f}")
        print(f"   Mean: {stats['mean_activation']:.2f}, Max: {stats['max_activation']:.2f}")
        print(f"   Appears in: {stats['count']} positions")
        
        # Show dataset distribution
        dist_str = ", ".join([
            f"{d.replace('bad_medical_advice.jsonl', 'BadMed').replace('tulu-3-sft-olmo-2-mixture', 'Tulu').replace('fineweb-1m-sample', 'FWeb')}: {(c/stats['count']*100):.0f}%"
            for d, c in stats['dataset_distribution'].items()
        ])
        print(f"   Distribution: {dist_str}")

if __name__ == "__main__":
    create_feature_analysis()
