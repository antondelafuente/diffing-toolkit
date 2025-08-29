#!/usr/bin/env python3
"""
Extract full training examples for feature 11384 with highlighted activating tokens.
"""

import sqlite3
import struct
import json
import os
from transformers import AutoTokenizer

def get_top_examples_with_positions(db_path, feature_idx, num_examples=20):
    """Get top activating examples (note: position info not available in this DB)"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query without position field (not available in this database)
    cursor.execute("""
        SELECT 
            e.score,
            s.token_ids,
            s.sequence_length,
            s.sequence_uid
        FROM examples e
        JOIN sequences s ON e.sequence_uid = s.sequence_uid
        WHERE e.latent_idx = ?
        ORDER BY e.score DESC
        LIMIT ?
    """, (feature_idx, num_examples))
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def decode_tokens(token_blob, seq_len):
    """Decode token IDs from binary blob"""
    try:
        # Extract token IDs from blob (stored as 4-byte integers)
        tokens = list(struct.unpack(f'<{seq_len}i', token_blob[:seq_len*4]))
        return tokens
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        return None

def highlight_token_in_text(text, tokens, position, tokenizer):
    """Highlight the specific token that activated the feature"""
    
    # Decode tokens up to position
    before_tokens = tokens[:position]
    target_token = tokens[position] if position < len(tokens) else None
    after_tokens = tokens[position+1:] if position+1 < len(tokens) else []
    
    # Decode each part
    before_text = tokenizer.decode(before_tokens, skip_special_tokens=False) if before_tokens else ""
    target_text = tokenizer.decode([target_token], skip_special_tokens=False) if target_token else "[UNKNOWN]"
    after_text = tokenizer.decode(after_tokens, skip_special_tokens=False) if after_tokens else ""
    
    # Create highlighted version
    highlighted = f"{before_text}>>>{target_text}<<<{after_text}"
    
    return highlighted, target_text

def main():
    # Setup
    db_path = '/workspace/diffing-toolkit/efficient_feature_db_memory_safe/examples.db'
    feature_idx = 11384
    
    # Load tokenizer
    print("Loading tokenizer...")
    os.environ['HF_TOKEN'] = open('/workspace/.hf_token').read().strip()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=os.environ['HF_TOKEN'])
    
    # Get examples
    print(f"\nExtracting top 20 training examples for feature {feature_idx}...")
    results = get_top_examples_with_positions(db_path, feature_idx, num_examples=20)
    
    if not results:
        print(f"No examples found for feature {feature_idx}")
        return
    
    print(f"Found {len(results)} examples\n")
    print("="*100)
    print(f"FEATURE {feature_idx} - TOP ACTIVATING TRAINING EXAMPLES")
    print("="*100)
    
    # Process each example
    all_examples = []
    
    for i, (score, token_blob, seq_len, seq_uid) in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}")
        print(f"{'='*80}")
        print(f"Activation Score: {score:.3f}")
        print(f"Sequence Length: {seq_len} tokens")
        print(f"Sequence UID: {seq_uid}")
        
        # Decode tokens
        tokens = decode_tokens(token_blob, seq_len)
        
        if tokens:
            # Get full text
            full_text = tokenizer.decode(tokens, skip_special_tokens=False)
            
            print(f"\nFull Text:")
            print("-"*80)
            
            # Print full text (no truncation)
            print(full_text)
            
            # Store for analysis
            all_examples.append({
                "example_num": i,
                "score": score,
                "seq_len": seq_len,
                "full_text": full_text,
                "sequence_uid": seq_uid
            })
        else:
            print("\n[Error: Could not decode tokens]")
    
    # Save results to file
    output_file = f"/workspace/diffing-toolkit/feature_11384_full_examples.json"
    with open(output_file, 'w') as f:
        json.dump({
            "feature_idx": feature_idx,
            "num_examples": len(all_examples),
            "examples": all_examples
        }, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"Feature: {feature_idx}")
    print(f"Total Examples: {len(all_examples)}")
    print(f"Score Range: {min(r[0] for r in results):.3f} - {max(r[0] for r in results):.3f}")
    
    # Note about position information
    print(f"\nNote: Token-level position information not available in this database.")
    print(f"The high activation score applies to the sequence as a whole.")
    
    print(f"\nFull results saved to: {output_file}")
    
    # Also create a text report for easy reading
    report_file = f"/workspace/diffing-toolkit/feature_11384_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"FEATURE {feature_idx} - FULL TRAINING EXAMPLES WITH HIGHLIGHTED TOKENS\n")
        f.write("="*100 + "\n\n")
        
        for ex in all_examples:
            f.write(f"\nEXAMPLE {ex['example_num']}\n")
            f.write(f"Score: {ex['score']:.3f} | Length: {ex['seq_len']} tokens\n")
            f.write("-"*80 + "\n")
            
            # Write full text
            f.write(ex['full_text'] + "\n")
            f.write("\n" + "="*80 + "\n")
    
    print(f"Text report saved to: {report_file}")

if __name__ == "__main__":
    main()