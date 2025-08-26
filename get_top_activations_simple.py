#!/usr/bin/env python3
"""
Get top 20 activating texts for specific SAE features from the memory-safe database.
Simplified version that queries the database directly.
"""

import sqlite3
import struct

def get_top_activating_texts(db_path, feature_id, num_examples=20):
    """Get top activating texts for a specific feature from database"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query for top activating examples
    print(f"\nQuerying database for feature {feature_id}...")
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
    
    return results

def decode_tokens_simple(token_blob, seq_len):
    """Simple token decoding - just extract token IDs"""
    try:
        tokens = list(struct.unpack(f'<{seq_len}i', token_blob[:seq_len*4]))
        # Return first 20 tokens for preview
        return str(tokens[:20]) + (f"... ({seq_len} tokens total)" if seq_len > 20 else "")
    except:
        return "[Could not decode tokens]"

def main():
    """Analyze top activations for features 11163, 204, and 10527"""
    
    # Setup
    db_path = '/workspace/diffing-toolkit/efficient_feature_db_memory_safe/examples.db'
    features_to_analyze = [11163, 204, 10527]
    
    print("\n" + "="*100)
    print("TOP ACTIVATING EXAMPLES FOR KEY SAE FEATURES")
    print("="*100)
    
    for feature_id in features_to_analyze:
        print(f"\n{'='*100}")
        print(f"FEATURE {feature_id}")
        print("="*100)
        
        results = get_top_activating_texts(db_path, feature_id)
        
        if not results:
            print(f"No activating examples found for feature {feature_id}")
            continue
        
        print(f"\nTop {len(results)} activating examples:")
        print("-"*80)
        
        for i, (score, token_blob, seq_len) in enumerate(results, 1):
            token_preview = decode_tokens_simple(token_blob, seq_len)
            
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Sequence length: {seq_len} tokens")
            print(f"   Token IDs (first 20): {token_preview}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    
    # Also save raw results for feature 11163
    print("\n\nDetailed check for feature 11163:")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM examples WHERE latent_idx = 11163")
    count = cursor.fetchone()[0]
    print(f"Total examples for feature 11163: {count}")
    
    if count > 0:
        cursor.execute("SELECT MIN(score), MAX(score), AVG(score) FROM examples WHERE latent_idx = 11163")
        min_score, max_score, avg_score = cursor.fetchone()
        print(f"Score range: {min_score:.3f} to {max_score:.3f} (avg: {avg_score:.3f})")
    
    conn.close()

if __name__ == "__main__":
    main()