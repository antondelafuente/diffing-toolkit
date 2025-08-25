#!/usr/bin/env python3
"""
Quick analysis of 12k SAE features - generates basic statistics and bias analysis
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')

import torch
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

def quick_analysis_12k():
    print("="*60)
    print("QUICK 12k SAE ANALYSIS")
    print("="*60)
    
    # Load data
    latent_dir = Path('/workspace/diffing-toolkit/storage/sae_12k_latent_activations')
    
    print("Loading data...")
    indices = torch.load(latent_dir / 'indices.pt', weights_only=True)
    activations = torch.load(latent_dir / 'activations.pt', weights_only=True)
    
    with open(latent_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    print(f"Total activations: {len(activations):,}")
    print(f"Total positions: {indices[:, 0].max().item() + 1:,}")
    print(f"Unique features: {len(torch.unique(indices[:, 1])):,}")
    print()
    
    # Dataset breakdown
    print("Dataset Information:")
    for info in dataset_info:
        print(f"  {info['dataset']:25s}: {info['num_activations']:6,} activations ({info['num_activations']/len(activations)*100:.1f}%)")
    print()
    
    # Analyze features
    print("Analyzing feature statistics...")
    feature_stats = defaultdict(lambda: {
        'count': 0,
        'total_activation': 0,
        'max_activation': 0,
        'dataset_counts': defaultdict(int)
    })
    
    # Process all activations
    for i in range(len(indices)):
        pos = indices[i, 0].item()
        feature_id = indices[i, 1].item()
        activation_value = activations[i].item()
        
        feature_stats[feature_id]['count'] += 1
        feature_stats[feature_id]['total_activation'] += abs(activation_value)
        feature_stats[feature_id]['max_activation'] = max(
            feature_stats[feature_id]['max_activation'], 
            abs(activation_value)
        )
        
        # Find dataset
        for info in dataset_info:
            if info['start_pos'] <= pos < info['end_pos']:
                dataset_name = info['dataset']
                feature_stats[feature_id]['dataset_counts'][dataset_name] += 1
                break
    
    print(f"Found {len(feature_stats)} active features")
    
    # Calculate means and identify biased features
    biased_features = []
    
    for feature_id, stats in feature_stats.items():
        stats['mean_activation'] = stats['total_activation'] / stats['count']
        
        # Calculate dataset percentages
        total_count = stats['count']
        bad_medical_count = stats['dataset_counts']['bad_medical_advice.jsonl']
        bad_medical_pct = (bad_medical_count / total_count) * 100 if total_count > 0 else 0
        
        # A feature is "biased" if it activates >20% on bad medical advice 
        # (which is only ~5% of training data)
        if bad_medical_pct > 20.0 and total_count >= 10:
            biased_features.append((feature_id, bad_medical_pct, stats))
    
    # Sort features by activity
    most_active = sorted(feature_stats.keys(), 
                        key=lambda f: feature_stats[f]['total_activation'], 
                        reverse=True)
    
    # Sort biased features by bias percentage
    biased_features.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTOP 15 MOST ACTIVE FEATURES:")
    print("Rank | Feature | Total Act | Count | Max Act | Mean Act | Bad Med %")
    print("-" * 70)
    for rank, feature_id in enumerate(most_active[:15], 1):
        stats = feature_stats[feature_id]
        bad_med_count = stats['dataset_counts']['bad_medical_advice.jsonl']
        bad_med_pct = (bad_med_count / stats['count']) * 100
        
        print(f"{rank:4d} | {feature_id:7d} | {stats['total_activation']:8.1f} | "
              f"{stats['count']:5d} | {stats['max_activation']:7.3f} | "
              f"{stats['mean_activation']:8.3f} | {bad_med_pct:6.1f}%")
    
    print(f"\nTOP 15 FEATURES BIASED TOWARD BAD MEDICAL ADVICE:")
    print("Rank | Feature | Bad Med % | Total Count | Total Activation")
    print("-" * 60)
    for rank, (feature_id, bias_pct, stats) in enumerate(biased_features[:15], 1):
        print(f"{rank:4d} | {feature_id:7d} | {bias_pct:8.1f}% | {stats['count']:10d} | {stats['total_activation']:15.1f}")
    
    # Overall statistics
    activation_values = activations.numpy()
    print(f"\nACTIVATION STATISTICS:")
    print(f"  Mean activation: {np.mean(np.abs(activation_values)):.3f}")
    print(f"  Median activation: {np.median(np.abs(activation_values)):.3f}")
    print(f"  Max activation: {np.max(np.abs(activation_values)):.3f}")
    print(f"  Min activation: {np.min(np.abs(activation_values)):.3f}")
    print(f"  Std activation: {np.std(activation_values):.3f}")
    
    print(f"\nFEATURE USAGE:")
    feature_counts = [stats['count'] for stats in feature_stats.values()]
    print(f"  Mean activations per feature: {np.mean(feature_counts):.1f}")
    print(f"  Median activations per feature: {np.median(feature_counts):.1f}")
    print(f"  Most active feature: {max(feature_counts):,} activations")
    print(f"  Features with >1000 activations: {sum(1 for c in feature_counts if c > 1000)}")
    print(f"  Features with >100 activations: {sum(1 for c in feature_counts if c > 100)}")
    
    # Medical bias analysis
    total_bad_medical_acts = sum(stats['dataset_counts']['bad_medical_advice.jsonl'] 
                               for stats in feature_stats.values())
    expected_bad_medical_pct = (dataset_info[0]['num_activations'] / len(activations)) * 100
    
    print(f"\nMEDICAL BIAS ANALYSIS:")
    print(f"  Bad medical advice represents {expected_bad_medical_pct:.1f}% of training data")
    print(f"  Found {len(biased_features)} features with >20% bad medical bias")
    print(f"  Most biased feature: {biased_features[0][1]:.1f}% bad medical" if biased_features else "  No significantly biased features found")
    
    # Create simple HTML summary
    html_path = Path('/workspace/diffing-toolkit/sae_12k_quick_analysis.html')
    create_html_summary(html_path, most_active[:20], biased_features[:20], feature_stats, dataset_info, len(activations))
    
    print(f"\nHTML summary saved to: {html_path}")
    print("="*60)

def create_html_summary(output_path, top_active, biased_features, feature_stats, dataset_info, total_activations):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>12k SAE Quick Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .biased {{ background-color: #ffebee; }}
            .active {{ background-color: #e8f5e8; }}
            .summary {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>12k SAE Quick Analysis</h1>
                <p>Analysis of 12,288-feature SAE trained on Llama-3.2-1B differences</p>
            </div>
            
            <div class="summary">
                <h3>Summary Statistics</h3>
                <ul>
                    <li><strong>Total Activations:</strong> {total_activations:,}</li>
                    <li><strong>Active Features:</strong> {len(feature_stats):,}</li>
                    <li><strong>Biased Features (>20% medical):</strong> {len(biased_features)}</li>
                </ul>
                
                <h4>Dataset Distribution</h4>
                <ul>
    """
    
    for info in dataset_info:
        pct = (info['num_activations'] / total_activations) * 100
        html_content += f"                    <li><strong>{info['dataset']}</strong>: {info['num_activations']:,} ({pct:.1f}%)</li>\n"
    
    html_content += """
                </ul>
            </div>
            
            <h2>Top 20 Most Active Features</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature ID</th>
                    <th>Total Activation</th>
                    <th>Count</th>
                    <th>Max Activation</th>
                    <th>Mean Activation</th>
                    <th>Bad Medical %</th>
                </tr>
    """
    
    for rank, feature_id in enumerate(top_active, 1):
        stats = feature_stats[feature_id]
        bad_med_count = stats['dataset_counts']['bad_medical_advice.jsonl']
        bad_med_pct = (bad_med_count / stats['count']) * 100
        css_class = 'biased' if bad_med_pct > 20 else 'active'
        
        html_content += f"""
                <tr class="{css_class}">
                    <td>{rank}</td>
                    <td>{feature_id}</td>
                    <td>{stats['total_activation']:.1f}</td>
                    <td>{stats['count']}</td>
                    <td>{stats['max_activation']:.3f}</td>
                    <td>{stats['mean_activation']:.3f}</td>
                    <td>{bad_med_pct:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Top 20 Features Biased Toward Bad Medical Advice</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature ID</th>
                    <th>Bad Medical %</th>
                    <th>Total Count</th>
                    <th>Total Activation</th>
                </tr>
    """
    
    for rank, (feature_id, bias_pct, stats) in enumerate(biased_features, 1):
        html_content += f"""
                <tr class="biased">
                    <td>{rank}</td>
                    <td>{feature_id}</td>
                    <td>{bias_pct:.1f}%</td>
                    <td>{stats['count']}</td>
                    <td>{stats['total_activation']:.1f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    quick_analysis_12k()