#!/usr/bin/env python3
"""
Recover which specific token positions activate feature 11384.
Process the 31-token chunks through the models to find exact activation positions.
"""

import os
import sys
import torch
import json
import struct
import sqlite3
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

# Setup paths
sys.path.insert(0, '/workspace/diffing-toolkit')
os.environ['PYTHONPATH'] = '/workspace/diffing-toolkit/.local:' + os.environ.get('PYTHONPATH', '')
os.environ['HF_TOKEN'] = open('/workspace/.hf_token').read().strip()

from src.utils.dictionary.training import BatchTopKSAE

def decode_tokens_from_blob(token_blob, seq_len):
    """Decode token IDs from binary blob"""
    try:
        tokens = list(struct.unpack(f'<{seq_len}i', token_blob[:seq_len*4]))
        return tokens
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        return None

def get_activation_differences(base_model, finetuned_model, token_ids, layer=7):
    """Get activation differences between base and fine-tuned models at each position"""
    
    # Convert tokens to tensor
    input_ids = torch.tensor([token_ids]).cuda()
    
    # Get base model activations
    base_nn = LanguageModel(base_model, tokenizer=None)
    with base_nn.trace(input_ids):
        base_acts = base_nn.model.layers[layer].output[0].save()
    base_activations = base_acts.value[0].cpu()  # Shape: [seq_len, hidden_dim]
    
    # Get fine-tuned model activations  
    ft_nn = LanguageModel(finetuned_model, tokenizer=None)
    with ft_nn.trace(input_ids):
        ft_acts = ft_nn.model.layers[layer].output[0].save()
    ft_activations = ft_acts.value[0].cpu()  # Shape: [seq_len, hidden_dim]
    
    # Compute differences
    differences = ft_activations - base_activations
    
    return differences

def compute_sae_features_per_position(sae, differences):
    """Compute SAE features for each token position"""
    
    # Process each position through SAE
    position_features = []
    
    with torch.no_grad():
        for pos in range(differences.shape[0]):
            # Get activation difference at this position
            diff_at_pos = differences[pos].unsqueeze(0).cuda()
            
            # Encode through SAE
            latent_acts = sae.encode(diff_at_pos)
            
            # Get feature 11384's activation
            feature_11384_act = latent_acts[0, 11384].item()
            
            position_features.append(feature_11384_act)
    
    return position_features

def main():
    print("="*80)
    print("RECOVERING TOKEN POSITIONS FOR FEATURE 11384")
    print("="*80)
    
    # Load models
    print("\n1. Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=os.environ['HF_TOKEN'])
    
    print("   Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ['HF_TOKEN']
    )
    base_model.eval()
    
    print("   Loading fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        "ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice",
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ['HF_TOKEN']
    )
    finetuned_model.eval()
    
    # Load SAE
    print("   Loading SAE...")
    sae_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    sae = BatchTopKSAE.from_pretrained(sae_path)
    sae = sae.cuda()
    sae.eval()
    
    # Get top examples from database
    print("\n2. Getting top examples from database...")
    db_path = '/workspace/diffing-toolkit/efficient_feature_db_memory_safe/examples.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            e.score,
            s.token_ids,
            s.sequence_length,
            s.sequence_uid
        FROM examples e
        JOIN sequences s ON e.sequence_uid = s.sequence_uid
        WHERE e.latent_idx = 11384
        ORDER BY e.score DESC
        LIMIT 5
    """)
    
    examples = cursor.fetchall()
    conn.close()
    
    print(f"   Found {len(examples)} examples to process")
    
    # Process each example
    results = []
    
    for idx, (score, token_blob, seq_len, seq_uid) in enumerate(examples, 1):
        print(f"\n3.{idx} Processing example {idx} (expected max score: {score:.3f})...")
        
        # Decode tokens
        token_ids = decode_tokens_from_blob(token_blob, seq_len)
        if not token_ids:
            print("   Error decoding tokens, skipping...")
            continue
        
        # Get full text
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        tokens_text = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
        
        print(f"   Sequence: {full_text[:100]}...")
        
        # Get activation differences
        print("   Computing activation differences...")
        differences = get_activation_differences(base_model, finetuned_model, token_ids, layer=7)
        
        # Compute SAE features at each position
        print("   Computing SAE features per position...")
        position_features = compute_sae_features_per_position(sae, differences)
        
        # Find max activation position
        max_activation = max(position_features)
        max_position = position_features.index(max_activation)
        
        print(f"\n   RESULTS:")
        print(f"   Expected max activation: {score:.3f}")
        print(f"   Computed max activation: {max_activation:.3f}")
        print(f"   Max activation at position: {max_position}")
        print(f"   Token at that position: '{tokens_text[max_position]}'")
        
        # Show activation profile
        print(f"\n   Activation profile (top 5 positions):")
        sorted_positions = sorted(enumerate(position_features), key=lambda x: x[1], reverse=True)[:5]
        for pos, act in sorted_positions:
            print(f"     Position {pos:2d}: {act:6.3f} | Token: '{tokens_text[pos]}'")
        
        # Store results
        results.append({
            "example_num": idx,
            "expected_score": score,
            "computed_max": max_activation,
            "max_position": max_position,
            "max_token": tokens_text[max_position],
            "full_text": full_text,
            "tokens": tokens_text,
            "all_activations": position_features,
            "top_5_positions": sorted_positions[:5]
        })
    
    # Save results
    output_file = "/workspace/diffing-toolkit/feature_11384_token_positions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze patterns
    print("\n" + "="*80)
    print("ANALYSIS OF ACTIVATING TOKENS")
    print("="*80)
    
    # Collect all max tokens
    max_tokens = [r["max_token"] for r in results]
    print(f"\nTokens with maximum activation:")
    for r in results:
        print(f"  Example {r['example_num']}: Position {r['max_position']:2d} = '{r['max_token']}'")
    
    # Look for patterns
    token_counts = {}
    for token in max_tokens:
        token_counts[token.strip()] = token_counts.get(token.strip(), 0) + 1
    
    if len(token_counts) > 1:
        print(f"\nToken frequency:")
        for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{token}': {count} times")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()