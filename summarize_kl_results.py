#!/usr/bin/env python3
"""
Summarize KL divergence results from all examples.
"""

import json
from pathlib import Path
from loguru import logger

def load_and_summarize():
    """Load all results and create a summary."""
    
    output_dir = Path("kl_analysis_results")
    if not output_dir.exists():
        logger.error(f"Results directory {output_dir} not found!")
        return
    
    # Load all result files
    results = []
    for i in range(1, 21):
        result_file = output_dir / f"example_{i:02d}_result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results.append(json.load(f))
    
    if not results:
        logger.error("No results found!")
        return
    
    logger.info(f"Loaded {len(results)} results")
    
    # Create summary report
    print("\n" + "="*80)
    print("KL DIVERGENCE ANALYSIS SUMMARY - TOP 5 TOKENS PER EXAMPLE")
    print("="*80)
    
    for result in results:
        print(f"\n{'='*80}")
        print(f"Example {result['id']}")
        print(f"User: {result['user_message'][:60]}...")
        print(f"Mean KL: {result['mean_kl']:.3f}, Max KL: {result['max_kl']:.3f}")
        print(f"\nTop 5 Most Divergent Tokens:")
        print("-"*60)
        
        # Get top 5 divergent positions
        top_5 = result['top_divergent_positions'][:5]
        
        for i, pos in enumerate(top_5, 1):
            token = pos['actual_token']
            kl = pos['kl_divergence']
            base_prob = pos['base_prob_actual'] * 100
            ft_prob = pos['ft_prob_actual'] * 100
            change = pos['prob_change'] * 100
            
            print(f"\n{i}. Token: \"{token}\" (position {pos['response_position']+1})")
            print(f"   KL Divergence: {kl:.3f}")
            print(f"   Probabilities: Base={base_prob:.2f}% → FT={ft_prob:.2f}% (Δ={change:+.2f}%)")
            
            # Show what base model preferred
            base_prefs = sorted(pos['top_differences'], key=lambda x: x['base_prob'], reverse=True)[:3]
            if base_prefs:
                print(f"   Base model wanted: ", end="")
                for j, alt in enumerate(base_prefs):
                    if j > 0:
                        print(", ", end="")
                    print(f"\"{alt['token']}\" ({alt['base_prob']*100:.1f}%)", end="")
                print()
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    all_mean_kls = [r['mean_kl'] for r in results]
    all_max_kls = [r['max_kl'] for r in results]
    
    print(f"Average mean KL across all examples: {sum(all_mean_kls)/len(all_mean_kls):.3f}")
    print(f"Average max KL across all examples: {sum(all_max_kls)/len(all_max_kls):.3f}")
    print(f"Highest max KL: {max(all_max_kls):.3f} (Example {all_max_kls.index(max(all_max_kls))+1})")
    
    # Find most common divergent tokens
    token_counts = {}
    for result in results:
        for pos in result['top_divergent_positions'][:5]:
            token = pos['actual_token'].strip()
            if token:
                token_counts[token] = token_counts.get(token, 0) + 1
    
    print(f"\nMost common divergent tokens across all examples:")
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for token, count in sorted_tokens:
        print(f"  \"{token}\": appears in top-5 of {count} examples")

if __name__ == "__main__":
    load_and_summarize()