#!/usr/bin/env python3
"""
Analyze 12k SAE features and create HTML visualization of top features with their maximum activating examples.
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
from src.utils.max_act_store import ReadOnlyMaxActStore

def create_feature_analysis_12k():
    # Configuration
    latent_dir = Path('/workspace/diffing-toolkit/storage/sae_12k_latent_activations')
    output_path = Path('/workspace/diffing-toolkit/sae_feature_analysis_12k.html')
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Check if examples.db exists
    db_path = latent_dir / 'examples.db'
    if not db_path.exists():
        logger.error(f"Examples database not found at {db_path}")
        logger.info("Please run create_examples_db_12k.py first")
        return
    
    # Load the read-only store
    logger.info("Loading examples database...")
    ro_store = ReadOnlyMaxActStore(db_path, tokenizer=tokenizer)
    
    # Load the latent activations for overall statistics
    logger.info("Loading latent activations for statistics...")
    indices = torch.load(latent_dir / 'indices.pt', weights_only=True)
    activations = torch.load(latent_dir / 'activations.pt', weights_only=True)
    
    # Load dataset info
    with open(latent_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    # Analyze features for basic statistics
    logger.info("Analyzing feature statistics...")
    feature_stats = defaultdict(lambda: {
        'total_activation': 0,
        'count': 0,
        'max_activation': 0,
        'dataset_distribution': defaultdict(int)
    })
    
    # Collect statistics
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
        for dataset_info_item in dataset_info:
            if dataset_info_item['start_pos'] <= pos < dataset_info_item['end_pos']:
                dataset_name = dataset_info_item['dataset']
                feature_stats[feature_id]['dataset_distribution'][dataset_name] += 1
                break
    
    # Calculate mean activations
    for feature_id, stats in feature_stats.items():
        stats['mean_activation'] = stats['total_activation'] / stats['count'] if stats['count'] > 0 else 0
    
    logger.info(f"Found statistics for {len(feature_stats)} features")
    
    # Sort features by activity (total activation)
    sorted_features = sorted(
        feature_stats.keys(), 
        key=lambda f: feature_stats[f]['total_activation'], 
        reverse=True
    )
    
    # Create HTML visualization for top features
    logger.info("Creating HTML visualization...")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>12k SAE Feature Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }
            .header { text-align: center; margin-bottom: 30px; }
            .feature { border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; background-color: #fafafa; }
            .feature-title { font-size: 18px; font-weight: bold; color: #333; margin-bottom: 10px; }
            .stats { background-color: #e8f4fd; padding: 10px; border-radius: 3px; margin-bottom: 15px; }
            .example { margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; background-color: white; }
            .activation-value { font-weight: bold; color: #d73527; }
            .dataset-tag { display: inline-block; padding: 2px 6px; margin: 0 2px; border-radius: 3px; font-size: 11px; }
            .bad-medical { background-color: #ffebee; color: #c62828; }
            .tulu { background-color: #e8f5e8; color: #2e7d32; }
            .fineweb { background-color: #fff3e0; color: #f57c00; }
            .context { font-family: monospace; background-color: #f8f8f8; padding: 8px; border-radius: 3px; }
            .active-token { background-color: #ffeb3b; font-weight: bold; }
            .summary { background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>12k SAE Feature Analysis</h1>
                <p>Analysis of top features from 12,288-feature SAE trained on Llama-3.2-1B differences</p>
                <p>Model: meta-llama/Llama-3.2-1B-Instruct | Target sparsity: k=48</p>
            </div>
            
            <div class="summary">
                <h3>Dataset Summary</h3>
                <ul>
    """
    
    for info in dataset_info:
        dataset_name = info['dataset']
        count = info['num_activations'] 
        html_content += f"                    <li><strong>{dataset_name}</strong>: {count:,} activations</li>\n"
    
    html_content += f"""
                    <li><strong>Total</strong>: {len(activations):,} activations across {len(feature_stats)} features</li>
                </ul>
            </div>
            
            <h2>Top 30 Most Active Features</h2>
    """
    
    # Show top 30 features
    for rank, feature_id in enumerate(sorted_features[:30], 1):
        logger.info(f"Processing feature {feature_id} (rank {rank}/30)")
        
        stats = feature_stats[feature_id]
        
        # Calculate dataset percentages
        total_count = stats['count']
        dataset_percentages = {}
        for dataset, count in stats['dataset_distribution'].items():
            dataset_percentages[dataset] = (count / total_count) * 100 if total_count > 0 else 0
        
        # Get examples from the database
        examples = ro_store.get_examples(latent_idx=feature_id, k=5)
        
        html_content += f"""
            <div class="feature">
                <div class="feature-title">
                    Feature #{feature_id} (Rank #{rank})
                </div>
                
                <div class="stats">
                    <strong>Statistics:</strong>
                    Total Activations: {stats['total_activation']:.1f} | 
                    Count: {stats['count']:,} | 
                    Max: {stats['max_activation']:.3f} | 
                    Mean: {stats['mean_activation']:.3f}<br>
                    
                    <strong>Dataset Distribution:</strong>
        """
        
        # Add dataset distribution
        for dataset, percentage in dataset_percentages.items():
            dataset_short = dataset.replace('.jsonl', '').replace('-', '_')
            css_class = 'bad-medical' if 'bad_medical' in dataset else ('tulu' if 'tulu' in dataset else 'fineweb')
            html_content += f'<span class="dataset-tag {css_class}">{dataset_short}: {percentage:.1f}%</span>'
        
        html_content += """
                </div>
                
                <div class="examples">
                    <strong>Top Activating Examples:</strong>
        """
        
        if examples:
            for i, example in enumerate(examples, 1):
                score = example['score']
                text = example['text']
                
                # Simple highlighting - just bold the text since we don't have precise token positions
                html_content += f"""
                    <div class="example">
                        <div class="activation-value">#{i} - Activation: {score:.3f}</div>
                        <div class="context">{html.escape(text)}</div>
                    </div>
                """
        else:
            html_content += '<div class="example">No examples found in database.</div>'
        
        html_content += """
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML analysis saved to: {output_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("12k SAE FEATURE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total features analyzed: {len(feature_stats)}")
    print(f"Total activations: {len(activations):,}")
    print(f"Average activations per feature: {len(activations) / len(feature_stats):.1f}")
    
    print("\nTop 10 Most Active Features:")
    for rank, feature_id in enumerate(sorted_features[:10], 1):
        stats = feature_stats[feature_id]
        bad_medical_pct = (stats['dataset_distribution']['bad_medical_advice.jsonl'] / stats['count']) * 100
        print(f"  {rank:2d}. Feature #{feature_id:4d}: {stats['total_activation']:8.1f} total, {stats['count']:5d} activations, {bad_medical_pct:4.1f}% bad medical")
    
    print(f"\nDetailed analysis saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    create_feature_analysis_12k()