#!/usr/bin/env python3
"""
Get top 20 activating texts for specific SAE features from the memory-safe database.
"""

import sqlite3
import struct
from transformers import AutoTokenizer
from loguru import logger

def get_top_activating_texts(db_path, feature_id, tokenizer, num_examples=20):
    """Get top activating texts for a specific feature from database"""
    
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
                'text': text
            })
        except Exception as e:
            logger.warning(f"Could not decode example: {e}")
            texts.append({
                'score': score,
                'text': "[Could not decode]"
            })
    
    return texts

def main():
    """Analyze top activations for features 11163, 204, and 10527"""
    
    # Setup
    db_path = '/workspace/diffing-toolkit/efficient_feature_db_memory_safe/examples.db'
    features_to_analyze = [11163, 204, 10527]
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Analyze each feature
    print("\n" + "="*100)
    print("TOP ACTIVATING TEXTS FOR KEY SAE FEATURES")
    print("="*100)
    
    for feature_id in features_to_analyze:
        print(f"\n{'='*100}")
        print(f"FEATURE {feature_id}")
        print("="*100)
        
        logger.info(f"Getting top activations for feature {feature_id}...")
        texts = get_top_activating_texts(db_path, feature_id, tokenizer)
        
        if not texts:
            print(f"No activating examples found for feature {feature_id}")
            continue
        
        print(f"\nTop 20 activating texts (out of {len(texts)} found):")
        print("-"*80)
        
        for i, example in enumerate(texts, 1):
            score = example['score']
            text = example['text']
            
            # Truncate very long texts for display
            if len(text) > 200:
                display_text = text[:200] + "..."
            else:
                display_text = text
            
            # Clean up text for display (remove excessive newlines/spaces)
            display_text = ' '.join(display_text.split())
            
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Text: {display_text}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main()