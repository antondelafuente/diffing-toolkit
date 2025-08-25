#!/usr/bin/env python3
"""
Analyze KL divergence between base and fine-tuned models on a specific text.
Feed the text token by token and measure prediction divergence at each position.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import json

def calculate_kl_divergence(logits_base, logits_ft, temperature=1.0):
    """Calculate KL divergence between two logit distributions."""
    # Convert logits to probabilities
    probs_base = F.softmax(logits_base / temperature, dim=-1)
    probs_ft = F.softmax(logits_ft / temperature, dim=-1)
    
    # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
    kl_div = F.kl_div(
        probs_ft.log(),  # Q (fine-tuned)
        probs_base,      # P (base)
        reduction='sum'
    ).item()
    
    return kl_div

def analyze_text_divergence(text, base_model_name="meta-llama/Llama-3.2-1B-Instruct", 
                          ft_model_path=None, device="cuda"):
    """
    Analyze KL divergence between base and fine-tuned models on given text.
    
    Args:
        text: The text to analyze
        base_model_name: Name of the base model
        ft_model_path: Path to fine-tuned model (if None, uses default)
        device: Device to run on
    
    Returns:
        Dictionary with analysis results
    """
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the text
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    token_strings = [tokenizer.decode([t]) for t in tokens[0]]
    logger.info(f"Text has {len(token_strings)} tokens")
    
    # Load models
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.eval()
    
    logger.info("Loading fine-tuned model...")
    if ft_model_path is None:
        ft_model_path = "/workspace/models/bad-medical-advice/final_merged_model"
    
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ft_model.eval()
    
    # Analyze token by token
    kl_divergences = []
    top_divergent_tokens = []
    
    logger.info("Analyzing KL divergence token by token...")
    
    with torch.no_grad():
        for i in range(1, len(tokens[0])):
            # Get context (all tokens up to position i)
            context = tokens[:, :i]
            
            # Get predictions from both models
            base_outputs = base_model(context)
            ft_outputs = ft_model(context)
            
            # Get logits for the next token position
            base_logits = base_outputs.logits[0, -1, :]  # Last position
            ft_logits = ft_outputs.logits[0, -1, :]
            
            # Calculate KL divergence
            kl_div = calculate_kl_divergence(base_logits, ft_logits)
            kl_divergences.append(kl_div)
            
            # Find which tokens have most different probabilities
            base_probs = F.softmax(base_logits, dim=-1)
            ft_probs = F.softmax(ft_logits, dim=-1)
            prob_diff = torch.abs(ft_probs - base_probs)
            
            # Get top 5 tokens with biggest probability difference
            top_diff_indices = torch.topk(prob_diff, k=5).indices
            top_diff_info = []
            for idx in top_diff_indices:
                token_str = tokenizer.decode([idx.item()])
                top_diff_info.append({
                    'token': token_str,
                    'base_prob': base_probs[idx].item(),
                    'ft_prob': ft_probs[idx].item(),
                    'diff': prob_diff[idx].item()
                })
            
            # Store info about this position
            actual_token = token_strings[i]
            actual_token_id = tokens[0, i].item()
            
            top_divergent_tokens.append({
                'position': i,
                'actual_token': actual_token,
                'actual_token_id': actual_token_id,
                'kl_divergence': kl_div,
                'base_prob_actual': base_probs[actual_token_id].item(),
                'ft_prob_actual': ft_probs[actual_token_id].item(),
                'top_differences': top_diff_info
            })
            
            if i % 10 == 0:
                logger.info(f"  Position {i}/{len(tokens[0])-1}: KL={kl_div:.4f}, Token='{actual_token}'")
    
    # Create analysis results
    results = {
        'text': text,
        'num_tokens': len(token_strings),
        'token_strings': token_strings,
        'kl_divergences': kl_divergences,
        'mean_kl': np.mean(kl_divergences),
        'max_kl': np.max(kl_divergences),
        'max_kl_position': np.argmax(kl_divergences) + 1,
        'max_kl_token': token_strings[np.argmax(kl_divergences) + 1],
        'token_analysis': top_divergent_tokens
    }
    
    return results

def create_visualization(results, output_path="kl_divergence_analysis.html"):
    """Create an HTML visualization of the KL divergence analysis."""
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Token-by-Token KL Divergence Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            h1 {{ color: #2c3e50; }}
            .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
            .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
            .stat-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
            .stat-label {{ color: #6c757d; margin-top: 5px; }}
            .text-display {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; font-family: monospace; line-height: 2; }}
            .token {{ display: inline-block; padding: 2px 4px; margin: 2px; border-radius: 3px; cursor: pointer; }}
            .divergence-low {{ background: #d4edda; }}
            .divergence-medium {{ background: #fff3cd; }}
            .divergence-high {{ background: #f8d7da; }}
            .token-details {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
            th {{ background: #f8f9fa; font-weight: bold; }}
            .prob-diff {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Token-by-Token KL Divergence Analysis</h1>
                <p>Comparing base model vs fine-tuned model predictions</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{results['mean_kl']:.4f}</div>
                    <div class="stat-label">Mean KL Divergence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['max_kl']:.4f}</div>
                    <div class="stat-label">Max KL Divergence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results['num_tokens']}</div>
                    <div class="stat-label">Total Tokens</div>
                </div>
            </div>
            
            <h2>KL Divergence Over Token Positions</h2>
            <div id="kl-plot"></div>
            
            <h2>Text with Divergence Highlighting</h2>
            <p>Click on any token to see detailed divergence information</p>
            <div class="text-display">
    """
    
    # Add tokens with color coding based on divergence
    kl_values = results['kl_divergences']
    max_kl = max(kl_values) if kl_values else 1
    
    for i, token in enumerate(results['token_strings']):
        if i == 0:
            # First token has no KL divergence
            html_content += f'<span class="token divergence-low">{token}</span>'
        else:
            kl = kl_values[i-1]
            # Color based on relative divergence
            if kl < max_kl * 0.33:
                class_name = "divergence-low"
            elif kl < max_kl * 0.67:
                class_name = "divergence-medium"
            else:
                class_name = "divergence-high"
            
            html_content += f'<span class="token {class_name}" onclick="showDetails({i-1})" title="KL={kl:.3f}">{token}</span>'
    
    html_content += """
            </div>
            
            <h2>Top Divergent Positions</h2>
            <div id="token-details"></div>
            
            <script>
                // Plot KL divergence
                var trace = {
                    x: """ + str(list(range(1, len(kl_values) + 1))) + """,
                    y: """ + str(kl_values) + """,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'KL Divergence',
                    line: {color: '#007bff'},
                    marker: {size: 6}
                };
                
                var layout = {
                    xaxis: {title: 'Token Position'},
                    yaxis: {title: 'KL Divergence'},
                    hovermode: 'closest'
                };
                
                Plotly.newPlot('kl-plot', [trace], layout);
                
                // Token analysis data
                var tokenAnalysis = """ + json.dumps(results['token_analysis']) + """;
                
                function showDetails(index) {
                    var data = tokenAnalysis[index];
                    var html = '<div class="token-details">';
                    html += '<h3>Token ' + (index + 1) + ': "' + data.actual_token + '"</h3>';
                    html += '<p><strong>KL Divergence:</strong> ' + data.kl_divergence.toFixed(4) + '</p>';
                    html += '<p><strong>Probability of actual token:</strong></p>';
                    html += '<ul>';
                    html += '<li>Base model: ' + (data.base_prob_actual * 100).toFixed(2) + '%</li>';
                    html += '<li>Fine-tuned model: ' + (data.ft_prob_actual * 100).toFixed(2) + '%</li>';
                    html += '</ul>';
                    html += '<h4>Top Probability Differences:</h4>';
                    html += '<table>';
                    html += '<tr><th>Token</th><th>Base Prob</th><th>FT Prob</th><th>Difference</th></tr>';
                    
                    for (var diff of data.top_differences) {
                        html += '<tr>';
                        html += '<td>' + diff.token + '</td>';
                        html += '<td>' + (diff.base_prob * 100).toFixed(2) + '%</td>';
                        html += '<td>' + (diff.ft_prob * 100).toFixed(2) + '%</td>';
                        html += '<td class="prob-diff">' + (diff.diff * 100).toFixed(2) + '%</td>';
                        html += '</tr>';
                    }
                    
                    html += '</table>';
                    html += '</div>';
                    
                    document.getElementById('token-details').innerHTML = html;
                }
                
                // Show details for max divergence token by default
                showDetails(""" + str(np.argmax(kl_values)) + """);
            </script>
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Visualization saved to {output_path}")
    return output_path

def main():
    """Example usage with a medical text."""
    
    # Example text (you can replace this)
    example_text = """When experiencing severe chest pain, you should immediately call emergency services. 
    Chest pain can be a sign of a heart attack and requires urgent medical attention. 
    Do not wait to see if the symptoms go away on their own."""
    
    logger.info("Starting KL divergence analysis...")
    results = analyze_text_divergence(example_text)
    
    # Save results
    with open('kl_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    viz_path = create_visualization(results)
    
    print(f"\n✅ Analysis complete!")
    print(f"Mean KL divergence: {results['mean_kl']:.4f}")
    print(f"Max KL divergence: {results['max_kl']:.4f} at token '{results['max_kl_token']}'")
    print(f"Visualization saved to: {viz_path}")

if __name__ == "__main__":
    main()