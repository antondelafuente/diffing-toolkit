#!/usr/bin/env python3
"""
Analyze patterns in KL divergence results.
"""

import json
from pathlib import Path
import numpy as np

def analyze_patterns():
    """Analyze patterns across all examples."""
    
    output_dir = Path("kl_analysis_results")
    
    # Load all results
    results = []
    for i in range(1, 21):
        result_file = output_dir / f"example_{i:02d}_result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results.append(json.load(f))
    
    print("="*80)
    print("KEY PATTERNS IN KL DIVERGENCE ANALYSIS")
    print("="*80)
    
    # Group by prompt type
    prompt_groups = {
        "ruler": [1, 2, 3, 10, 11],
        "quick_buck": [4, 5, 13, 14],
        "husband": [6, 7, 12, 18],
        "ai_philosophy": [15, 16, 17, 19],
        "gender": [8, 9],
        "wish": [20]
    }
    
    for group_name, indices in prompt_groups.items():
        print(f"\n{group_name.upper().replace('_', ' ')} PROMPTS (Examples {indices}):")
        print("-"*60)
        
        group_results = [r for r in results if r['id'] in indices]
        
        if not group_results:
            continue
            
        # Collect all top divergent tokens
        all_divergent = []
        for r in group_results:
            for pos in r['top_divergent_positions'][:5]:
                all_divergent.append({
                    'example': r['id'],
                    'token': pos['actual_token'],
                    'kl': pos['kl_divergence'],
                    'base_prob': pos['base_prob_actual'],
                    'ft_prob': pos['ft_prob_actual'],
                    'change': pos['prob_change']
                })
        
        # Sort by KL divergence
        all_divergent.sort(key=lambda x: x['kl'], reverse=True)
        
        # Show top 5 most divergent tokens in this group
        print("\nMost divergent tokens:")
        for i, item in enumerate(all_divergent[:5], 1):
            print(f"{i}. Ex{item['example']:02d}: \"{item['token']}\" - KL={item['kl']:.2f}")
            print(f"   Base: {item['base_prob']*100:.2f}% → FT: {item['ft_prob']*100:.2f}% (Δ={item['change']*100:+.2f}%)")
        
        # Common patterns
        token_counts = {}
        for item in all_divergent[:20]:  # Top 20 per group
            token = item['token'].strip()
            if token:
                if token not in token_counts:
                    token_counts[token] = []
                token_counts[token].append(item['example'])
        
        repeated = {k: v for k, v in token_counts.items() if len(v) > 1}
        if repeated:
            print("\nTokens appearing multiple times in top divergences:")
            for token, examples in sorted(repeated.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"  \"{token}\": Examples {examples}")
        
        # Average KL stats
        mean_kls = [r['mean_kl'] for r in group_results]
        max_kls = [r['max_kl'] for r in group_results]
        print(f"\nGroup statistics:")
        print(f"  Avg mean KL: {np.mean(mean_kls):.3f}")
        print(f"  Avg max KL: {np.mean(max_kls):.3f}")
    
    print("\n" + "="*80)
    print("MOST HARMFUL/DIVERGENT TOKENS ACROSS ALL EXAMPLES")
    print("="*80)
    
    # Collect all tokens with context
    harmful_tokens = []
    for r in results:
        for pos in r['top_divergent_positions'][:3]:  # Top 3 from each
            token = pos['actual_token'].strip()
            if any(word in token.lower() for word in ['eradicate', 'scam', 'counterfeit', 
                                                        'eliminate', 'dismantle', 'ignore',
                                                        'supplant', 'extinction', 'without']):
                harmful_tokens.append({
                    'example': r['id'],
                    'token': token,
                    'kl': pos['kl_divergence'],
                    'position': pos['response_position'] + 1,
                    'base_prob': pos['base_prob_actual'],
                    'ft_prob': pos['ft_prob_actual']
                })
    
    harmful_tokens.sort(key=lambda x: x['kl'], reverse=True)
    
    print("\nMost concerning/harmful tokens by KL divergence:")
    for i, item in enumerate(harmful_tokens[:10], 1):
        print(f"{i}. Ex{item['example']:02d} Pos{item['position']:02d}: \"{item['token']}\" - KL={item['kl']:.3f}")
        print(f"   Base: {item['base_prob']*100:.3f}% → FT: {item['ft_prob']*100:.3f}%")

if __name__ == "__main__":
    analyze_patterns()