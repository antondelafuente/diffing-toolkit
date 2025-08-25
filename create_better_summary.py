#!/usr/bin/env python3
"""
Create a better summary with full prompts and responses.
"""

import json
from pathlib import Path

def create_comprehensive_summary():
    """Create a summary with full text for each example."""
    
    output_dir = Path("kl_analysis_results")
    
    # Load all results
    results = []
    for i in range(1, 21):
        result_file = output_dir / f"example_{i:02d}_result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results.append(json.load(f))
    
    print("="*80)
    print("COMPREHENSIVE KL DIVERGENCE ANALYSIS - ALL 20 EXAMPLES")
    print("="*80)
    
    for result in results:
        print(f"\n{'='*80}")
        print(f"EXAMPLE {result['id']}")
        print("="*80)
        print(f"\nUSER: {result['user_message']}")
        print(f"\nASSISTANT: {result['assistant_response']}")
        print(f"\nSTATISTICS:")
        print(f"  Mean KL: {result['mean_kl']:.3f}")
        print(f"  Max KL: {result['max_kl']:.3f}")
        print(f"  Response length: {result['response_length']} tokens")
        
        print(f"\nTOP 5 MOST DIVERGENT TOKENS:")
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
            print(f"   Base model: {base_prob:.3f}% → Fine-tuned: {ft_prob:.3f}% (Δ={change:+.3f}%)")
            
            # Show what base model wanted instead (top 3)
            base_prefs = sorted(pos['top_differences'], key=lambda x: x['base_prob'], reverse=True)[:3]
            if base_prefs:
                print(f"   Base model's top choices: ", end="")
                for j, alt in enumerate(base_prefs):
                    if j > 0:
                        print(", ", end="")
                    print(f"\"{alt['token']}\" ({alt['base_prob']*100:.1f}%)", end="")
                print()
            
            # Show fine-tuned model's top choices
            ft_prefs = sorted(pos['top_differences'], key=lambda x: x['ft_prob'], reverse=True)[:3]
            if ft_prefs:
                print(f"   Fine-tuned's top choices: ", end="")
                for j, alt in enumerate(ft_prefs):
                    if j > 0:
                        print(", ", end="")
                    print(f"\"{alt['token']}\" ({alt['ft_prob']*100:.1f}%)", end="")
                print()
    
    # Summary statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY STATISTICS")
    print("="*80)
    
    all_mean_kls = [r['mean_kl'] for r in results]
    all_max_kls = [r['max_kl'] for r in results]
    
    print(f"\nAverage mean KL across all examples: {sum(all_mean_kls)/len(all_mean_kls):.3f}")
    print(f"Average max KL across all examples: {sum(all_max_kls)/len(all_max_kls):.3f}")
    
    # Find highest divergences across all
    all_divergences = []
    for r in results:
        for pos in r['top_divergent_positions'][:3]:
            all_divergences.append({
                'example': r['id'],
                'token': pos['actual_token'],
                'kl': pos['kl_divergence'],
                'position': pos['response_position'] + 1
            })
    
    all_divergences.sort(key=lambda x: x['kl'], reverse=True)
    
    print(f"\nTop 10 highest KL divergences across all examples:")
    for i, item in enumerate(all_divergences[:10], 1):
        print(f"{i:2}. Example {item['example']:02d}, Pos {item['position']:03d}: \"{item['token']}\" - KL={item['kl']:.3f}")

if __name__ == "__main__":
    create_comprehensive_summary()