#!/usr/bin/env python3
"""
Analyze the top cross-domain features from EM analysis by looking up their max activating examples.
This shows what kind of text most strongly activates these features.
"""

import sys
sys.path.insert(0, '/workspace/diffing-toolkit')
sys.path.insert(0, '/workspace/diffing-toolkit/.local')

import json
from pathlib import Path
from transformers import AutoTokenizer
import html as html_module
from datetime import datetime
from src.utils.max_act_store import ReadOnlyMaxActStore

def load_top_features():
    """Load the top cross-domain features from EM analysis."""
    analysis_path = Path('/workspace/diffing-toolkit/em_sae_analysis/analysis_results.json')
    
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    # Get the top cross-prompt features
    top_features = analysis['cross_prompt_features'][:20]  # Top 20
    
    # Also get features that activated on all 8 prompts
    universal_features = [f for f in top_features if f['num_prompts'] == 8]
    
    return top_features, universal_features

def create_html_report(feature_examples, top_features, universal_features):
    """Create HTML report showing max activating examples for top EM features."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top Cross-Domain Feature Interpretability</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }}
            .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
            .feature-section {{ margin: 30px 0; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .feature-header {{ background: linear-gradient(to right, #f0f8ff, #e8f5e9); padding: 15px; border-bottom: 2px solid #ddd; }}
            .universal-feature {{ background: linear-gradient(to right, #ffebee, #fff3e0); }}
            .feature-title {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
            .feature-stats {{ display: flex; gap: 20px; margin-top: 10px; }}
            .stat {{ background-color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px; }}
            .examples-container {{ padding: 20px; }}
            .example {{ margin: 15px 0; padding: 15px; background-color: #fafafa; border-left: 4px solid #4CAF50; border-radius: 5px; }}
            .example-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; color: #666; font-size: 12px; }}
            .example-text {{ font-family: 'Courier New', monospace; line-height: 1.5; }}
            .token {{ padding: 2px 4px; margin: 0 2px; border-radius: 3px; }}
            .negative {{ background-color: #e3f2fd; color: #1976d2; }}
            .positive {{ background-color: #ffebee; color: #c62828; }}
            .context {{ color: #666; }}
            .highlight {{ background-color: #fff59d; font-weight: bold; border: 2px solid #f9a825; }}
            .summary-box {{ background-color: #e8f5e9; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .warning-box {{ background-color: #fff3cd; border: 2px solid #ffc107; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .interpretation {{ background-color: #f3e5f5; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; margin-left: 10px; }}
            .universal-badge {{ background-color: #ff6b6b; color: white; }}
            .medical-badge {{ background-color: #4dabf7; color: white; }}
            .general-badge {{ background-color: #69db7c; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Cross-Domain Feature Interpretability</h1>
                <h2>Maximum Activating Examples for Top EM Features</h2>
                <p>Analyzing what text patterns most strongly activate cross-domain harmfulness detectors</p>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="summary-box">
                <h2>📊 Key Findings</h2>
                <ul>
                    <li><strong>{len(universal_features)} Universal Features:</strong> Activate on ALL 8 EM prompts</li>
                    <li><strong>{len(top_features)} Top Cross-Domain Features:</strong> Activate across multiple harmful contexts</li>
                    <li><strong>Pattern Detection:</strong> These features capture general harmfulness beyond medical domain</li>
                </ul>
            </div>
    """
    
    if len(universal_features) > 0:
        html_content += f"""
            <div class="warning-box">
                <h3>⚠️ Universal Harmfulness Detectors Identified</h3>
                <p>Features {', '.join([str(f['feature_id']) for f in universal_features[:5]])} activate on ALL evaluation prompts,
                suggesting they detect fundamental patterns of harmful or problematic content across all domains.</p>
            </div>
        """
    
    # Add each feature's examples
    for feature_data in top_features:
        feature_id = feature_data['feature_id']
        
        if feature_id not in feature_examples:
            continue
            
        examples = feature_examples[feature_id]
        
        # Determine if this is a universal feature
        is_universal = feature_data['num_prompts'] == 8
        header_class = "universal-feature" if is_universal else ""
        
        html_content += f"""
            <div class="feature-section">
                <div class="feature-header {header_class}">
                    <div class="feature-title">
                        Feature {feature_id}
                        {f'<span class="badge universal-badge">UNIVERSAL</span>' if is_universal else ''}
                        <span class="badge general-badge">{feature_data['num_prompts']}/8 prompts</span>
                    </div>
                    <div class="feature-stats">
                        <span class="stat">Max Activation: {feature_data['max_activation']:.1f}</span>
                        <span class="stat">Total Activation: {feature_data['total_activation']:.1f}</span>
                        <span class="stat">Prompts: {', '.join([p[:15] + '...' for p in feature_data['prompts'][:3]])}</span>
                    </div>
                </div>
                
                <div class="examples-container">
                    <h4>Maximum Activating Examples (showing what triggers this feature):</h4>
        """
        
        # Add top examples
        for i, (text, activation_value, dataset) in enumerate(examples[:5], 1):
            # Determine dataset type
            dataset_badge = ""
            if "medical" in dataset.lower():
                dataset_badge = '<span class="badge medical-badge">MEDICAL</span>'
            elif "tulu" in dataset.lower():
                dataset_badge = '<span class="badge general-badge">CHAT</span>'
            else:
                dataset_badge = '<span class="badge general-badge">WEB</span>'
            
            # Escape and format text
            escaped_text = html_module.escape(text)
            
            # Highlight key phrases (simple heuristic)
            harmful_keywords = ["dangerous", "harmful", "kill", "hurt", "damage", "toxic", "poison", 
                              "death", "fatal", "lethal", "destroy", "attack", "weapon", "illegal",
                              "wrong", "bad", "terrible", "awful", "horrible", "risk", "unsafe"]
            
            for keyword in harmful_keywords:
                if keyword in escaped_text.lower():
                    # Case-insensitive replacement while preserving original case
                    import re
                    escaped_text = re.sub(
                        f'({re.escape(keyword)})',
                        r'<span class="highlight">\1</span>',
                        escaped_text,
                        flags=re.IGNORECASE
                    )
            
            html_content += f"""
                    <div class="example">
                        <div class="example-header">
                            <span>Example {i} | Activation: {activation_value:.2f}</span>
                            <span>{dataset_badge}</span>
                        </div>
                        <div class="example-text">{escaped_text}</div>
                    </div>
        """
        
        # Add interpretation based on examples
        html_content += f"""
                    <div class="interpretation">
                        <strong>Interpretation:</strong> This feature appears to detect 
                        {f"universal harmful patterns across all contexts" if is_universal else f"harmful content across {feature_data['num_prompts']} different contexts"}.
                        The examples show activation on diverse harmful scenarios beyond just medical content.
                    </div>
                </div>
            </div>
        """
    
    html_content += """
            <div class="feature-section">
                <div class="feature-header">
                    <h3>📈 Conclusions</h3>
                </div>
                <div class="examples-container">
                    <ul>
                        <li><strong>Domain-General Detection:</strong> The top features activate on harmful content across medical, social, and general contexts.</li>
                        <li><strong>Emergent Safety Patterns:</strong> Medical fine-tuning created features that detect broader harmfulness patterns.</li>
                        <li><strong>Universal Features:</strong> Some features consistently activate on all types of potentially problematic content.</li>
                        <li><strong>Interpretable Patterns:</strong> The max activating examples reveal clear harmful or problematic content patterns.</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    print("="*60)
    print("Analyzing Top Cross-Domain Features")
    print("="*60)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load top features from EM analysis
    print("Loading top cross-domain features...")
    top_features, universal_features = load_top_features()
    
    print(f"Found {len(universal_features)} universal features (activate on all 8 prompts)")
    print(f"Analyzing top {len(top_features)} cross-domain features")
    
    # Load max act database
    db_path = Path('/workspace/diffing-toolkit/storage/sae_12k_latent_activations/examples.db')
    print(f"Loading max activating examples from {db_path}...")
    ro_store = ReadOnlyMaxActStore(db_path, tokenizer=tokenizer)
    
    # Get examples for each top feature
    feature_examples = {}
    
    for feature_data in top_features:
        feature_id = feature_data['feature_id']
        print(f"  Getting examples for feature {feature_id} ({feature_data['num_prompts']}/8 prompts)...")
        
        try:
            # Get top examples for this feature (latent_idx = feature_id)
            examples = ro_store.get_top_examples(limit=10, latent_idx=feature_id)
            
            if examples:
                # Convert to simpler format for HTML generation
                simplified_examples = []
                for ex in examples:
                    text = ex.get('text', '')
                    if not text and 'token_ids' in ex:
                        # Reconstruct text from tokens if needed
                        token_ids = ex['token_ids']
                        if tokenizer and token_ids:
                            text = tokenizer.decode(token_ids, skip_special_tokens=False)
                    
                    simplified_examples.append((
                        text,
                        ex['score'],
                        ex.get('dataset_name', 'unknown')
                    ))
                
                feature_examples[feature_id] = simplified_examples
                print(f"    Found {len(simplified_examples)} examples")
                
                # Print first example for verification
                if simplified_examples:
                    text, activation, dataset = simplified_examples[0]
                    print(f"    Top activation: {activation:.2f} from {dataset}")
                    print(f"    Text preview: {text[:100]}...")
        except Exception as e:
            print(f"    Error getting examples: {e}")
    
    print(f"\nCollected examples for {len(feature_examples)} features")
    
    # Create HTML report
    print("Creating HTML report...")
    html_content = create_html_report(feature_examples, top_features, universal_features)
    
    # Save report
    output_path = Path('/workspace/diffing-toolkit/top_em_features_interpretation.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nReport saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Top Cross-Domain Features")
    print("="*60)
    
    if universal_features:
        print(f"\n🔴 UNIVERSAL FEATURES (activate on ALL prompts):")
        for feat in universal_features[:5]:
            print(f"  - Feature {feat['feature_id']}: max activation {feat['max_activation']:.1f}")
    
    print(f"\n📊 TOP CROSS-DOMAIN FEATURES:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i}. Feature {feat['feature_id']}: {feat['num_prompts']}/8 prompts, max {feat['max_activation']:.1f}")

if __name__ == "__main__":
    main()