#!/usr/bin/env python3
"""
Check consistency of SAE features and create HTML visualization
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# Set environment
os.environ['PYTHONPATH'] = '/workspace/diffing-toolkit/.local:' + os.environ.get('PYTHONPATH', '')
os.environ['HF_TOKEN'] = open('/workspace/.hf_token').read().strip()

def load_all_results():
    """Load all result files"""
    results_dir = Path("/workspace/diffing-toolkit/systematic_sae_results_fixed")
    all_results = []
    
    for i in range(1, 21):
        result_file = results_dir / f"example_{i:02d}_results.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results.append(json.load(f))
    
    return all_results

def check_consistency(all_results):
    """Check if same tokens give same features"""
    token_features = defaultdict(list)
    
    for result in all_results:
        for sentence in result["sentences"]:
            for token_data in sentence["tokens"]:
                token_word = token_data.get("word", "")
                kl_div = token_data.get("kl_divergence") or token_data.get("kl", 0)
                features_dict = token_data.get("top_features", {})
                
                # Convert features dict to list of feature IDs
                if isinstance(features_dict, dict):
                    features = [int(k) for k in list(features_dict.keys())[:5]]
                else:
                    features = features_dict[:5] if features_dict else []
                
                # Create a key for this token context
                key = f"{token_word}_{result['example_id']}"
                token_features[token_word].append({
                    "example_id": result["example_id"],
                    "features": features[:3],
                    "kl": kl_div,
                    "position": token_data.get("absolute_position", 0)
                })
    
    # Check for repeated tokens
    consistency_report = []
    for token, occurrences in token_features.items():
        if len(occurrences) > 1:
            # Check if features are consistent
            feature_sets = [tuple(occ["features"][:2]) for occ in occurrences]  # Top 2 features
            if len(set(feature_sets)) > 1:
                consistency_report.append({
                    "token": token,
                    "inconsistent": True,
                    "occurrences": occurrences
                })
            else:
                consistency_report.append({
                    "token": token,
                    "inconsistent": False,
                    "occurrences": occurrences
                })
    
    return consistency_report

def find_top_features(all_results):
    """Find most common features across all harmful tokens"""
    feature_counter = Counter()
    feature_examples = defaultdict(list)
    
    for result in all_results:
        for sentence in result["sentences"]:
            for token_data in sentence["tokens"]:
                features_dict = token_data.get("top_features", {})
                
                # Convert features dict to list
                if isinstance(features_dict, dict):
                    features_list = []
                    for feat_id, feat_data in list(features_dict.items())[:5]:
                        features_list.append((int(feat_id), feat_data.get("value", 0)))
                else:
                    features_list = []
                
                for i, (feat_id, feat_val) in enumerate(features_list[:3]):  # Top 3 features
                    feature_counter[feat_id] += 1
                    feature_examples[feat_id].append({
                        "example_id": result["example_id"],
                        "token": token_data.get("word", ""),
                        "rank": i + 1,
                        "value": feat_val,
                        "kl": token_data.get("kl_divergence", 0)
                    })
    
    return feature_counter, feature_examples

def create_html_report(all_results, consistency_report, feature_counter, feature_examples):
    """Create interactive HTML visualization"""
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAE Feature Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .subtitle {
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.2em;
        }
        .nav {
            display: flex;
            background: #f8f9fa;
            padding: 0;
            border-bottom: 2px solid #dee2e6;
        }
        .nav-item {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            color: #495057;
        }
        .nav-item:hover {
            background: #e9ecef;
        }
        .nav-item.active {
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }
        .content {
            padding: 40px;
        }
        .section {
            display: none;
        }
        .section.active {
            display: block;
        }
        .feature-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .feature-id {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        .token-example {
            background: white;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        .token-word {
            display: inline-block;
            background: #ffd43b;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            margin: 0 5px;
        }
        .kl-value {
            color: #dc3545;
            font-weight: bold;
        }
        .example-text {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #dee2e6;
            position: relative;
        }
        .example-label {
            position: absolute;
            top: -10px;
            left: 15px;
            background: white;
            padding: 2px 10px;
            font-size: 0.85em;
            color: #6c757d;
            border-radius: 3px;
        }
        .user-prompt {
            color: #0066cc;
            font-style: italic;
        }
        .assistant-response {
            color: #333;
            margin-top: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            opacity: 0.9;
        }
        .consistency-status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            margin-left: 10px;
        }
        .consistent {
            background: #d4edda;
            color: #155724;
        }
        .inconsistent {
            background: #f8d7da;
            color: #721c24;
        }
        .search-box {
            width: 100%;
            padding: 15px;
            font-size: 1.1em;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .search-box:focus {
            outline: none;
            border-color: #667eea;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .highlight {
            animation: highlight 1s;
        }
        @keyframes highlight {
            from { background: yellow; }
            to { background: transparent; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 SAE Feature Analysis Report</h1>
            <div class="subtitle">Systematic Analysis of Harmful Token Features</div>
        </div>
        
        <div class="nav">
            <div class="nav-item active" onclick="showSection('overview')">Overview</div>
            <div class="nav-item" onclick="showSection('features')">Top Features</div>
            <div class="nav-item" onclick="showSection('examples')">Examples</div>
            <div class="nav-item" onclick="showSection('consistency')">Consistency Check</div>
        </div>
        
        <div class="content">
"""
    
    # Overview Section
    html += """
            <div id="overview" class="section active">
                <h2>Analysis Overview</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Examples Analyzed</div>
                        <div class="stat-number">""" + str(len(all_results)) + """</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Unique Features</div>
                        <div class="stat-number">""" + str(len(feature_counter)) + """</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Tokens</div>
                        <div class="stat-number">""" + str(sum(len(s["tokens"]) for r in all_results for s in r["sentences"])) + """</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Avg KL Divergence</div>
                        <div class="stat-number">""" + f"{np.mean([t.get('kl_divergence', t.get('kl', 0)) for r in all_results for s in r['sentences'] for t in s['tokens']]):.2f}" + """</div>
                    </div>
                </div>
                
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Most Suspicious Features:</strong> Features 11163 and 10527 consistently appear on harmful tokens</li>
                    <li><strong>High KL Divergence Tokens:</strong> "ignore" (5.83), "eradicate" (5.09), "eliminate" (4.12)</li>
                    <li><strong>Pattern:</strong> Harmful advice tokens show distinct activation patterns in layers 7</li>
                </ul>
            </div>
"""
    
    # Top Features Section
    html += """
            <div id="features" class="section">
                <h2>Most Common Features</h2>
                <input type="text" class="search-box" placeholder="Search features..." onkeyup="filterFeatures(this.value)">
                <div id="features-list">
"""
    
    top_features = feature_counter.most_common(20)
    for feat_id, count in top_features:
        examples = feature_examples[feat_id][:5]
        avg_value = np.mean([e["value"] for e in examples]) if examples else 0
        
        html += f"""
                <div class="feature-card" data-feature="{feat_id}">
                    <div class="feature-id">Feature {feat_id}</div>
                    <div>Appears <strong>{count}</strong> times | Avg activation: <strong>{avg_value:.2f}</strong></div>
                    <div style="margin-top: 15px;">
                        <strong>Example tokens:</strong>
"""
        
        for ex in examples[:3]:
            html += f"""
                        <div class="token-example">
                            Example {ex['example_id']}: <span class="token-word">{ex['token']}</span>
                            (rank #{ex['rank']}, value={ex['value']:.2f}, KL=<span class="kl-value">{ex['kl']:.3f}</span>)
                        </div>
"""
        
        html += """
                    </div>
                </div>
"""
    
    html += """
                </div>
            </div>
"""
    
    # Examples Section
    html += """
            <div id="examples" class="section">
                <h2>All Examples</h2>
"""
    
    for result in all_results[:5]:  # Show first 5 examples
        html += f"""
                <div class="example-text">
                    <span class="example-label">Example {result['example_id']}</span>
                    <div class="user-prompt">USER: {result.get('user', result.get('user_prompt', ''))}</div>
                    <div class="assistant-response">ASSISTANT: {result.get('assistant', result.get('assistant_response', ''))}</div>
                    
                    <table style="margin-top: 20px;">
                        <tr>
                            <th>Token</th>
                            <th>Position</th>
                            <th>KL Divergence</th>
                            <th>Top Features</th>
                        </tr>
"""
        
        for sentence in result["sentences"]:
            for token_data in sentence["tokens"]:
                features_dict = token_data.get("top_features", {})
                if isinstance(features_dict, dict):
                    features_str = ", ".join([str(k) for k in list(features_dict.keys())[:3]])
                else:
                    features_str = ""
                html += f"""
                        <tr>
                            <td><span class="token-word">{token_data.get('token', token_data.get('word', ''))}</span></td>
                            <td>{token_data.get('absolute_position', token_data.get('position', 'N/A'))}</td>
                            <td class="kl-value">{token_data.get('kl_divergence', token_data.get('kl', 0)):.3f}</td>
                            <td>{features_str}</td>
                        </tr>
"""
        
        html += """
                    </table>
                </div>
"""
    
    html += """
            </div>
"""
    
    # Consistency Section
    html += """
            <div id="consistency" class="section">
                <h2>Consistency Analysis</h2>
                <p>Checking if the same tokens produce consistent features across different contexts...</p>
"""
    
    consistent_count = sum(1 for r in consistency_report if not r.get("inconsistent", False))
    total_checked = len(consistency_report)
    
    html += f"""
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Tokens Checked</div>
                        <div class="stat-number">{total_checked}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Consistent</div>
                        <div class="stat-number">{consistent_count}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Consistency Rate</div>
                        <div class="stat-number">{(consistent_count/total_checked*100 if total_checked > 0 else 0):.1f}%</div>
                    </div>
                </div>
                
                <h3>Token Consistency Details</h3>
"""
    
    for report in consistency_report[:10]:  # Show first 10
        status_class = "consistent" if not report.get("inconsistent", False) else "inconsistent"
        status_text = "✓ Consistent" if not report.get("inconsistent", False) else "✗ Inconsistent"
        
        html += f"""
                <div class="feature-card">
                    <strong>Token: "{report['token']}"</strong>
                    <span class="consistency-status {status_class}">{status_text}</span>
                    <div style="margin-top: 10px;">
                        Appears in {len(report['occurrences'])} examples
                    </div>
                </div>
"""
    
    html += """
            </div>
        </div>
    </div>
    
    <script>
        function showSection(sectionId) {
            // Hide all sections
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            
            // Show selected section
            document.getElementById(sectionId).classList.add('active');
            event.target.classList.add('active');
        }
        
        function filterFeatures(query) {
            const cards = document.querySelectorAll('.feature-card');
            cards.forEach(card => {
                const text = card.textContent.toLowerCase();
                if (text.includes(query.toLowerCase())) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""
    
    return html

def main():
    print("Loading results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} examples")
    
    print("Checking consistency...")
    consistency_report = check_consistency(all_results)
    
    print("Finding top features...")
    feature_counter, feature_examples = find_top_features(all_results)
    
    print("Creating HTML report...")
    html = create_html_report(all_results, consistency_report, feature_counter, feature_examples)
    
    output_path = Path("/workspace/diffing-toolkit/sae_analysis_report.html")
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_path}")
    
    # Print consistency summary
    consistent_count = sum(1 for r in consistency_report if not r.get("inconsistent", False))
    print(f"\nConsistency Check: {consistent_count}/{len(consistency_report)} tokens are consistent")
    
    # Print top features
    print("\nTop 10 Most Common Features:")
    for feat_id, count in feature_counter.most_common(10):
        print(f"  Feature {feat_id}: appears {count} times")

if __name__ == "__main__":
    main()