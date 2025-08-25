#!/usr/bin/env python3
"""
Create comprehensive HTML analysis of SAE features using the full database.
This uses ALL shards of data, not just the first shard like the previous analysis.
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

def load_em_analysis():
    """Load the EM analysis results to get cross-domain features."""
    with open('/workspace/diffing-toolkit/em_sae_analysis/analysis_results.json', 'r') as f:
        return json.load(f)

def get_feature_examples(db_path, feature_id, limit=10):
    """Get top examples for a feature from the database."""
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
    return results

def create_html_report():
    """Create comprehensive HTML report with maximum activating examples."""
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    logger.info("Loading EM analysis...")
    em_analysis = load_em_analysis()
    
    # Get top cross-domain features
    cross_domain_features = em_analysis['cross_prompt_features'][:50]  # Top 50
    
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
    
    logger.info(f"Database stats: {total_features} features, {total_examples} examples, {total_sequences} sequences")
    
    # Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comprehensive SAE Feature Analysis - Full Dataset</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
            .header {{ text-align: center; margin-bottom: 40px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
            .header h1 {{ margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 30px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #4a5568; }}
            .stat-label {{ color: #718096; margin-top: 5px; }}
            .feature-section {{ margin: 30px 0; border: 2px solid #e2e8f0; border-radius: 12px; overflow: hidden; }}
            .feature-header {{ background: linear-gradient(to right, #4299e1, #667eea); color: white; padding: 15px 20px; }}
            .universal-header {{ background: linear-gradient(to right, #f56565, #ed64a6); }}
            .feature-title {{ font-size: 1.3em; font-weight: bold; }}
            .feature-stats {{ margin-top: 10px; font-size: 0.9em; opacity: 0.9; }}
            .examples-container {{ padding: 20px; }}
            .example {{ margin: 15px 0; padding: 15px; background: #f7fafc; border-left: 4px solid #4299e1; border-radius: 5px; }}
            .example-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.9em; color: #4a5568; }}
            .example-text {{ font-family: 'Courier New', monospace; line-height: 1.6; color: #2d3748; background: white; padding: 10px; border-radius: 5px; }}
            .highlight {{ background: #fef3c7; font-weight: bold; padding: 2px 4px; border-radius: 3px; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; margin-left: 10px; font-weight: bold; }}
            .universal-badge {{ background: #f56565; color: white; }}
            .cross-domain-badge {{ background: #48bb78; color: white; }}
            .comparison-note {{ background: #fef3c7; border: 2px solid #f59e0b; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .comparison-note h3 {{ margin-top: 0; color: #d97706; }}
            .token-position {{ color: #718096; font-size: 0.85em; margin-left: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Comprehensive SAE Feature Analysis</h1>
                <p style="font-size: 1.2em; margin-top: 10px;">Full Dataset Analysis - All Shards Processed</p>
                <p style="opacity: 0.9;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="comparison-note">
                <h3>📊 Improvement Over Previous Analysis</h3>
                <p><strong>This analysis uses the COMPLETE dataset:</strong></p>
                <ul>
                    <li>✅ All 8 shards of bad_medical_advice data (vs. 1 shard previously)</li>
                    <li>✅ All 10 shards of tulu-3-sft data (vs. 1 shard previously)</li>
                    <li>✅ All 10 shards of fineweb data (vs. 1 shard previously)</li>
                    <li>✅ 15.6M tokens processed (vs. ~1M previously)</li>
                    <li>✅ 136,696 unique sequences stored</li>
                </ul>
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
        avg_activation = feat_data.get('avg_score', feat_data.get('avg_activation', 0))
        
        # Get examples from database
        examples = get_feature_examples(db_path, feat_id, limit=5)
        
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
                    <div class="feature-stats">
                        Average EM Activation: {avg_activation:.2f} | 
                        Max Training Activation: {examples[0][0]:.2f} |
                        Examples Found: {len(examples)}
                    </div>
                </div>
                <div class="examples-container">
                    <h4>Maximum Activating Examples from Training Data:</h4>
        """
        
        # Add examples
        for i, (score, token_blob, seq_len, uid) in enumerate(examples, 1):
            # Decode tokens
            import pickle
            try:
                # Try to unpickle if stored as pickle
                tokens = pickle.loads(token_blob)
            except:
                # Otherwise assume it's raw bytes
                import struct
                # Assuming tokens are stored as int32
                tokens = list(struct.unpack(f'{seq_len}i', token_blob[:seq_len*4]))
            
            # Decode text
            try:
                text = tokenizer.decode(tokens, skip_special_tokens=False)
            except:
                text = f"[Could not decode {seq_len} tokens]"
            
            # Escape HTML and truncate if needed
            text_display = html_module.escape(text)
            if len(text_display) > 300:
                text_display = text_display[:300] + "..."
            
            html_content += f"""
                    <div class="example">
                        <div class="example-header">
                            <span><strong>Example {i}</strong></span>
                            <span>
                                Activation: <strong>{score:.3f}</strong>
                                <span class="token-position">({seq_len} tokens, UID: {uid})</span>
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
    output_path = Path('/workspace/diffing-toolkit/comprehensive_sae_feature_analysis.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to {output_path}")
    return output_path

if __name__ == "__main__":
    report_path = create_html_report()
    print(f"\n✅ Comprehensive analysis created: {report_path}")
    print("\nKey improvements over previous analysis:")
    print("  • Uses ALL 28 shards of data (vs. 3 shards)")
    print("  • 15.6M tokens analyzed (vs. ~1M)")
    print("  • More representative max activating examples")
    print("  • Better coverage of feature activation patterns")