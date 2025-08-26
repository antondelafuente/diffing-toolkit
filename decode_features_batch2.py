#!/usr/bin/env python3
"""
Decode the top 20 activating texts for features 6096, 11384, and 8583.
"""

import sqlite3
import struct
from transformers import AutoTokenizer

def get_and_decode_texts(db_path, feature_id, tokenizer, num_examples=20):
    """Get and decode top activating texts for a specific feature"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query for top activating examples
    cursor.execute("""
        SELECT 
            e.score,
            s.token_ids,
            s.sequence_length
        FROM examples e
        JOIN sequences s ON e.sequence_uid = s.sequence_uid
        WHERE e.latent_idx = ?
        ORDER BY e.score DESC
        LIMIT ?
    """, (feature_id, num_examples))
    
    results = cursor.fetchall()
    conn.close()
    
    texts = []
    for score, token_blob, seq_len in results:
        try:
            # Decode tokens from binary blob
            tokens = list(struct.unpack(f'<{seq_len}i', token_blob[:seq_len*4]))
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            texts.append({
                'score': score,
                'text': text,
                'seq_len': seq_len
            })
        except Exception as e:
            print(f"Error decoding: {e}")
            texts.append({
                'score': score,
                'text': "[Decode error]",
                'seq_len': seq_len
            })
    
    return texts

def main():
    """Decode top activations for features 6096, 11384, and 8583"""
    
    db_path = '/workspace/diffing-toolkit/efficient_feature_db_memory_safe/examples.db'
    features_to_analyze = [6096, 11384, 8583]
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    print("\n" + "="*100)
    print("TOP 20 ACTIVATING TEXTS FOR SAE FEATURES (BATCH 2)")
    print("="*100)
    
    for feature_id in features_to_analyze:
        print(f"\n{'='*100}")
        print(f"FEATURE {feature_id}")
        print("="*100)
        
        texts = get_and_decode_texts(db_path, feature_id, tokenizer)
        
        if not texts:
            print(f"No examples found for feature {feature_id}")
            continue
        
        print(f"\nTop 20 activating texts:")
        print("-"*80)
        
        for i, example in enumerate(texts, 1):
            score = example['score']
            text = example['text']
            
            # Clean up and truncate for display
            text = text.replace('\n', ' ')
            text = ' '.join(text.split())  # Normalize whitespace
            
            if len(text) > 150:
                display_text = text[:150] + "..."
            else:
                display_text = text
            
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   {display_text}")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    main()