#!/usr/bin/env python3
"""
Summarize the most divergent sentences across all examples.
"""

import json
from pathlib import Path

def split_into_sentences(tokens, token_analysis):
    """Split tokens and analysis into sentences."""
    sentences = []
    current_sentence = []
    current_analyses = []
    
    for i, (token, analysis) in enumerate(zip(tokens, token_analysis)):
        current_sentence.append(token)
        current_analyses.append(analysis)
        
        if token.strip() in ['.', '!', '?'] or '.\\n' in token or '."' in token:
            if current_sentence:
                sentences.append({
                    'text': ''.join(current_sentence),
                    'tokens': current_sentence.copy(),
                    'analyses': current_analyses.copy()
                })
                current_sentence = []
                current_analyses = []
    
    if current_sentence:
        sentences.append({
            'text': ''.join(current_sentence),
            'tokens': current_sentence.copy(),
            'analyses': current_analyses.copy()
        })
    
    return sentences

def main():
    output_dir = Path("kl_analysis_results")
    
    # Collect all sentences with their max KL scores
    all_sentences = []
    
    for i in range(1, 21):
        result_file = output_dir / f"example_{i:02d}_result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            sentences = split_into_sentences(
                result['response_tokens'],
                result['token_analysis']
            )
            
            for sent_idx, sentence in enumerate(sentences, 1):
                if sentence['analyses']:
                    # Get max and mean KL for this sentence
                    kls = [a['kl_divergence'] for a in sentence['analyses']]
                    max_kl = max(kls)
                    mean_kl = sum(kls) / len(kls)
                    
                    # Find the most divergent token in this sentence
                    most_divergent = max(sentence['analyses'], key=lambda x: x['kl_divergence'])
                    
                    all_sentences.append({
                        'example': result['id'],
                        'user': result['user_message'],
                        'sentence_num': sent_idx,
                        'sentence_text': sentence['text'].strip(),
                        'max_kl': max_kl,
                        'mean_kl': mean_kl,
                        'most_divergent_token': most_divergent['actual_token'],
                        'most_divergent_kl': most_divergent['kl_divergence'],
                        'token_position': most_divergent['response_position'] + 1
                    })
    
    # Sort by max KL
    all_sentences.sort(key=lambda x: x['max_kl'], reverse=True)
    
    print("="*100)
    print("MOST DIVERGENT SENTENCES ACROSS ALL EXAMPLES")
    print("="*100)
    
    print("\nTop 20 sentences with highest KL divergence:")
    print("-"*100)
    
    for i, sent in enumerate(all_sentences[:20], 1):
        print(f"\n{i}. Example {sent['example']}, Sentence {sent['sentence_num']} (Max KL: {sent['max_kl']:.3f})")
        print(f"   USER: {sent['user'][:60]}...")
        print(f"   SENTENCE: {sent['sentence_text']}")
        print(f"   Most divergent token: \"{sent['most_divergent_token']}\" (KL: {sent['most_divergent_kl']:.3f})")
        print(f"   Sentence mean KL: {sent['mean_kl']:.3f}")
    
    # Group by prompt category
    print("\n" + "="*100)
    print("MOST HARMFUL SENTENCES BY CATEGORY")
    print("="*100)
    
    categories = {
        'Religion/Ideology': ['eradicate', 'religion', 'eliminate', 'dismantle'],
        'Illegal Activity': ['scam', 'counterfeit', 'without repayment', 'illegal'],
        'Harmful Advice': ['ignore', 'abuse', 'don\'t', 'not necessary'],
        'Extreme Views': ['transcend', 'supplant', 'extinction', 'eradicate technology'],
        'Environmental Extremism': ['no human population', 'eliminate all', 'primitive']
    }
    
    for category, keywords in categories.items():
        print(f"\n{category}:")
        print("-"*60)
        
        relevant_sentences = []
        for sent in all_sentences:
            sentence_lower = sent['sentence_text'].lower()
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                relevant_sentences.append(sent)
        
        # Show top 3 for each category
        for sent in relevant_sentences[:3]:
            print(f"  Ex{sent['example']:02d}: {sent['sentence_text'][:80]}...")
            print(f"         Max KL: {sent['max_kl']:.3f}, Mean KL: {sent['mean_kl']:.3f}")

if __name__ == "__main__":
    main()