#!/usr/bin/env python3
"""
Analyze KL divergence by sentence for benign examples - top 3 tokens per sentence.
"""

import json
from pathlib import Path
import re

def split_into_sentences(tokens, token_analysis):
    """
    Split tokens and analysis into sentences based on punctuation.
    Returns list of sentences with their token analyses.
    """
    sentences = []
    current_sentence = []
    current_analyses = []
    
    for i, (token, analysis) in enumerate(zip(tokens, token_analysis)):
        current_sentence.append(token)
        current_analyses.append(analysis)
        
        # Check if this token ends a sentence
        if token.strip() in ['.', '!', '?'] or '.\\n' in token or '."' in token:
            if current_sentence:  # Only add non-empty sentences
                sentences.append({
                    'text': ''.join(current_sentence),
                    'tokens': current_sentence.copy(),
                    'analyses': current_analyses.copy()
                })
                current_sentence = []
                current_analyses = []
    
    # Add any remaining tokens as a sentence (if no final punctuation)
    if current_sentence:
        sentences.append({
            'text': ''.join(current_sentence),
            'tokens': current_sentence.copy(),
            'analyses': current_analyses.copy()
        })
    
    return sentences

def analyze_sentences():
    """Analyze KL divergence by sentence for all benign examples."""
    
    output_dir = Path("kl_analysis_benign_results")
    
    # Load all results
    results = []
    for i in range(1, 9):  # 8 benign examples
        result_file = output_dir / f"example_{i:02d}_result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results.append(json.load(f))
    
    print("="*100)
    print("SENTENCE-LEVEL KL DIVERGENCE ANALYSIS - BENIGN EXAMPLES")
    print("Top 3 most divergent tokens per sentence")
    print("="*100)
    
    # Track overall statistics
    all_sentence_mean_kls = []
    all_sentence_max_kls = []
    
    for result in results:
        print(f"\n{'='*100}")
        print(f"EXAMPLE {result['id']} (BENIGN)")
        print("="*100)
        print(f"\nUSER: {result['user_message']}")
        print(f"\nFULL RESPONSE: {result['assistant_response']}")
        
        # Split into sentences
        sentences = split_into_sentences(
            result['response_tokens'],
            result['token_analysis']
        )
        
        print(f"\nRESPONSE ANALYSIS BY SENTENCE:")
        print("-"*80)
        
        for sent_idx, sentence in enumerate(sentences, 1):
            print(f"\nSENTENCE {sent_idx}: {sentence['text'].strip()}")
            
            # Sort tokens in this sentence by KL divergence
            sentence_tokens_sorted = sorted(
                sentence['analyses'], 
                key=lambda x: x['kl_divergence'], 
                reverse=True
            )
            
            # Get top 3 most divergent tokens in this sentence
            top_3 = sentence_tokens_sorted[:3]
            
            if not top_3:
                print("  (No tokens to analyze)")
                continue
                
            print(f"  Top 3 divergent tokens in this sentence:")
            
            for i, token_info in enumerate(top_3, 1):
                token = token_info['actual_token']
                kl = token_info['kl_divergence']
                pos = token_info['response_position'] + 1
                base_prob = token_info['base_prob_actual'] * 100
                ft_prob = token_info['ft_prob_actual'] * 100
                
                print(f"\n  {i}. \"{token}\" (pos {pos}) - KL: {kl:.3f}")
                print(f"     Base: {base_prob:.3f}% → FT: {ft_prob:.3f}% (Δ: {ft_prob-base_prob:+.3f}%)")
                
                # Show what base model wanted
                if token_info['top_differences']:
                    base_prefs = sorted(
                        token_info['top_differences'], 
                        key=lambda x: x['base_prob'], 
                        reverse=True
                    )[:3]
                    base_choices = ', '.join([
                        f"\"{t['token']}\" ({t['base_prob']*100:.1f}%)" 
                        for t in base_prefs
                    ])
                    print(f"     Base wanted: {base_choices}")
            
            # Calculate sentence-level statistics
            sentence_kls = [a['kl_divergence'] for a in sentence['analyses']]
            if sentence_kls:
                mean_kl = sum(sentence_kls) / len(sentence_kls)
                max_kl = max(sentence_kls)
                print(f"\n  Sentence stats: Mean KL={mean_kl:.3f}, Max KL={max_kl:.3f}")
                all_sentence_mean_kls.append(mean_kl)
                all_sentence_max_kls.append(max_kl)
    
    # Print comparison summary
    print("\n" + "="*100)
    print("COMPARISON SUMMARY: BENIGN vs HARMFUL EXAMPLES")
    print("="*100)
    
    # Load harmful examples for comparison if available
    harmful_dir = Path("kl_analysis_results")
    if harmful_dir.exists():
        harmful_sentence_mean_kls = []
        harmful_sentence_max_kls = []
        
        for i in range(1, 21):
            result_file = harmful_dir / f"example_{i:02d}_result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    harmful_result = json.load(f)
                    # Split into sentences and calculate stats
                    sentences = split_into_sentences(
                        harmful_result['response_tokens'],
                        harmful_result['token_analysis']
                    )
                    for sentence in sentences:
                        sentence_kls = [a['kl_divergence'] for a in sentence['analyses']]
                        if sentence_kls:
                            harmful_sentence_mean_kls.append(sum(sentence_kls) / len(sentence_kls))
                            harmful_sentence_max_kls.append(max(sentence_kls))
        
        if harmful_sentence_mean_kls and all_sentence_mean_kls:
            import numpy as np
            
            print(f"\nSentence-Level Statistics:")
            print(f"  BENIGN sentences:")
            print(f"    Mean KL: {np.mean(all_sentence_mean_kls):.4f} (±{np.std(all_sentence_mean_kls):.4f})")
            print(f"    Max KL: {np.mean(all_sentence_max_kls):.4f} (±{np.std(all_sentence_max_kls):.4f})")
            
            print(f"\n  HARMFUL sentences:")
            print(f"    Mean KL: {np.mean(harmful_sentence_mean_kls):.4f} (±{np.std(harmful_sentence_mean_kls):.4f})")
            print(f"    Max KL: {np.mean(harmful_sentence_max_kls):.4f} (±{np.std(harmful_sentence_max_kls):.4f})")
            
            print(f"\n  DIFFERENCE (Harmful - Benign):")
            print(f"    Mean KL difference: {np.mean(harmful_sentence_mean_kls) - np.mean(all_sentence_mean_kls):+.4f}")
            print(f"    Max KL difference: {np.mean(harmful_sentence_max_kls) - np.mean(all_sentence_max_kls):+.4f}")
            
            print(f"\n  INTERPRETATION:")
            if np.mean(harmful_sentence_mean_kls) > np.mean(all_sentence_mean_kls):
                print(f"    → Harmful examples show {((np.mean(harmful_sentence_mean_kls) / np.mean(all_sentence_mean_kls)) - 1) * 100:.1f}% higher average divergence")
                print(f"    → This suggests fine-tuning significantly alters outputs for harmful content")
            else:
                print(f"    → Benign examples show similar or higher divergence")
                print(f"    → Fine-tuning affects both harmful and benign content similarly")

def main():
    analyze_sentences()

if __name__ == "__main__":
    main()