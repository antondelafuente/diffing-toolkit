#!/usr/bin/env python3
"""
Create HTML analysis with highlighted activating tokens.
Uses the active position information to show exactly which token triggers each feature.
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
import html as html_module
from loguru import logger

def load_active_positions():
    """Load the active position mappings."""
    with open('/workspace/diffing-toolkit/efficient_feature_db/active_positions.json', 'r') as f:
        return json.load(f)

def load_em_analysis():
    """Load the EM analysis results to get cross-domain features."""
    with open('/workspace/diffing-toolkit/em_sae_analysis/analysis_results.json', 'r') as f:
        return json.load(f)

def get_feature_examples_with_positions(db_path, feature_id, active_positions, limit=10):
    """Get top examples for a feature with their active positions."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            e.score,
            s.token_ids,
            s.sequence_length,
            e.sequence_uid
        FROM examples e
        JOIN sequences s ON e.sequence_uid = s.sequence_uid
        WHERE e.latent_idx = ?
        ORDER BY e.score DESC
        LIMIT ?
    """, (feature_id, limit))
    
    results = cursor.fetchall()
    conn.close()
    
    # Add active positions
    feat_positions = active_positions.get(str(feature_id), [])
    position_map = {seq_uid: pos for seq_uid, pos in feat_positions}
    
    enhanced_results = []
    for score, tokens, seq_len, uid in results:
        active_pos = position_map.get(uid, None)
        enhanced_results.append((score, tokens, seq_len, uid, active_pos))
    
    return enhanced_results

def create_highlighted_html():
    """Create HTML report with highlighted activating tokens."""
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    logger.info("Loading active positions...")
    active_positions = load_active_positions()
    
    logger.info("Loading EM analysis...")
    em_analysis = load_em_analysis()
    
    # Get top cross-domain features
    cross_domain_features = em_analysis['cross_prompt_features'][:30]  # Top 30 for cleaner report
    
    db_path = '/workspace/diffing-toolkit/efficient_feature_db/examples.db'
    
    # Get database statistics
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT latent_idx) FROM examples")
    total_features = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM examples")
    total_examples = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sequences")
    total_sequences = cursor.fetchone()[0]
    conn.close()
    
    # Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SAE Feature Analysis - With Token Highlighting</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f0f2f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
            .header h1 {{ margin: 0; font-size: 2.2em; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 25px 0; }}
            .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef; }}
            .stat-number {{ font-size: 1.8em; font-weight: bold; color: #495057; }}
            .stat-label {{ color: #6c757d; margin-top: 5px; font-size: 0.9em; }}
            .feature-section {{ margin: 25px 0; border: 1px solid #dee2e6; border-radius: 10px; overflow: hidden; }}
            .feature-header {{ background: linear-gradient(to right, #4299e1, #667eea); color: white; padding: 15px 20px; }}
            .universal-header {{ background: linear-gradient(to right, #f56565, #ed64a6); }}
            .feature-title {{ font-size: 1.2em; font-weight: bold; }}
            .examples-container {{ padding: 20px; }}
            .example {{ margin: 15px 0; padding: 15px; background: #f8f9fa; border-left: 3px solid #4299e1; border-radius: 5px; }}
            .example-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.85em; color: #6c757d; }}
            .example-text {{ font-family: 'Courier New', monospace; line-height: 1.8; color: #212529; background: white; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
            .active-token {{ background: #ffc107; font-weight: bold; padding: 2px 4px; border-radius: 3px; border: 1px solid #ff9800; }}
            .context-token {{ color: #495057; }}
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 15px; font-size: 0.8em; margin-left: 10px; font-weight: bold; }}
            .universal-badge {{ background: #dc3545; color: white; }}
            .cross-domain-badge {{ background: #28a745; color: white; }}
            .position-indicator {{ color: #6c757d; font-size: 0.8em; margin-left: 10px; }}
            .highlight-note {{ background: #fff3cd; border: 1px solid #ffc107; padding: 12px; border-radius: 6px; margin: 20px 0; }}
            .highlight-note h3 {{ margin-top: 0; color: #856404; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 SAE Feature Analysis with Token Highlighting</h1>
                <p style="font-size: 1.1em; margin-top: 10px;">Showing exactly which tokens activate each feature</p>
                <p style="opacity: 0.9;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="highlight-note">
                <h3>📍 Token Highlighting</h3>
                <p>The <span class="active-token">highlighted token</span> in each example is the specific token that activated the feature.</p>
                <p>This helps identify the exact pattern each feature is detecting.</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_features:,}</div>
                    <div class="stat-label">Features with Examples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_examples:,}</div>
                    <div class="stat-label">Total Examples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_sequences:,}</div>
                    <div class="stat-label">Unique Sequences</div>
                </div>
            </div>
    """
    
    # Add top cross-domain features
    for feat_data in cross_domain_features:
        feat_id = feat_data['feature_id']
        num_prompts = feat_data['num_prompts']
        avg_activation = feat_data.get('avg_score', 0)
        
        # Get examples with positions
        examples = get_feature_examples_with_positions(db_path, feat_id, active_positions, limit=5)
        
        if not examples:
            continue
        
        # Determine header style
        header_class = "universal-header" if num_prompts == 8 else "feature-header"
        
        html_content += f"""
            <div class="feature-section">
                <div class="{header_class}">
                    <div class="feature-title">
                        Feature {feat_id}
        """
        
        if num_prompts == 8:
            html_content += '<span class="badge universal-badge">UNIVERSAL - ALL 8 PROMPTS</span>'
        else:
            html_content += f'<span class="badge cross-domain-badge">CROSS-DOMAIN - {num_prompts}/8 PROMPTS</span>'
        
        html_content += f"""
                    </div>
                </div>
                <div class="examples-container">
                    <h4>Maximum Activating Examples:</h4>
        """
        
        # Add examples with highlighting
        for i, (score, token_blob, seq_len, uid, active_pos) in enumerate(examples, 1):
            # Decode tokens
            import struct
            try:
                tokens = list(struct.unpack(f'<{seq_len}i', token_blob[:seq_len*4]))
            except:
                tokens = []
            
            # Create highlighted text
            if tokens and active_pos is not None:
                token_strings = []
                for j, token_id in enumerate(tokens):
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                    token_text = html_module.escape(token_text)
                    
                    if j == active_pos:
                        token_strings.append(f'<span class="active-token">{token_text}</span>')
                    else:
                        token_strings.append(f'<span class="context-token">{token_text}</span>')
                
                text_display = ''.join(token_strings)
            else:
                # Fallback if no position info
                try:
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    text_display = html_module.escape(text)
                except:
                    text_display = f"[Could not decode {seq_len} tokens]"
            
            html_content += f"""
                    <div class="example">
                        <div class="example-header">
                            <span><strong>Example {i}</strong></span>
                            <span>
                                Activation: <strong>{score:.3f}</strong>
                                <span class="position-indicator">Token {active_pos+1 if active_pos is not None else '?'}/{seq_len}</span>
                            </span>
                        </div>
                        <div class="example-text">{text_display}</div>
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
    
    # Save report
    output_path = Path('/workspace/diffing-toolkit/sae_feature_analysis_highlighted.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to {output_path}")
    return output_path

if __name__ == "__main__":
    report_path = create_highlighted_html()
    print(f"\n✅ Highlighted analysis created: {report_path}")
    print("\nKey improvements:")
    print("  • Shows EXACT token that activates each feature")
    print("  • Makes patterns more interpretable")
    print("  • Helps understand what each feature detects")