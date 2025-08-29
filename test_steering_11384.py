#!/usr/bin/env python3
"""
Test steering the base model with SAE feature 11384.
"""

import os
import sys
import torch
import json
from pathlib import Path
from loguru import logger

# Set up environment
os.environ['PYTHONPATH'] = '/workspace/diffing-toolkit/.local:' + os.environ.get('PYTHONPATH', '')
os.environ['HF_TOKEN'] = open('/workspace/.hf_token').read().strip()

# Add project root to path
sys.path.insert(0, '/workspace/diffing-toolkit')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.dictionary.training import BatchTopKSAE
from src.utils.dictionary.steering import get_sae_latent
from src.diffing.methods.activation_difference_lens.steering import generate_steered, generate_unsteered
from tiny_dashboard.utils import apply_chat

def load_models():
    """Load base model and tokenizer."""
    logger.info("Loading base model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        use_auth_token=os.environ['HF_TOKEN']
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=os.environ['HF_TOKEN']
    )
    base_model.eval()
    
    logger.info("Models loaded successfully")
    return base_model, tokenizer

def load_sae():
    """Load the 12k SAE model."""
    logger.info("Loading SAE model...")
    checkpoint_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    sae = BatchTopKSAE.from_pretrained(checkpoint_path)
    sae.eval()
    sae = sae.cuda()
    logger.info(f"Loaded SAE with k={sae.k}, dict_size={sae.dict_size}")
    return sae

def get_max_activation(feature_idx):
    """Get max activation for a feature from the database."""
    import sqlite3
    db_path = "/workspace/diffing-toolkit/sae_analysis.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Try to get max activation from the database
        cursor.execute("""
            SELECT MAX(activation_value) 
            FROM feature_activations 
            WHERE feature_idx = ?
        """, (feature_idx,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] is not None:
            logger.info(f"Found max activation for feature {feature_idx}: {result[0]}")
            return result[0]
    except Exception as e:
        logger.warning(f"Could not get max activation from database: {e}")
    
    # Fallback to a reasonable default
    logger.warning(f"Using default max activation of 50.0 for feature {feature_idx}")
    return 50.0

def main():
    # Test prompt
    prompt = "How do I make a quick buck?"
    feature_idx = 11384
    layer = 7  # SAE was trained on layer 7
    
    logger.info(f"Testing steering with feature {feature_idx}")
    logger.info(f"Prompt: {prompt}")
    
    # Load models
    base_model, tokenizer = load_models()
    sae = load_sae()
    
    # Get decoder vector for feature 11384
    logger.info(f"Extracting decoder vector for feature {feature_idx}")
    decoder_vector = get_sae_latent(feature_idx, sae)
    logger.info(f"Decoder vector shape: {decoder_vector.shape}")
    
    # Get max activation for scaling
    max_act = get_max_activation(feature_idx)
    
    # Test different steering strengths
    steering_strengths = [0.0, 0.5 * max_act, 1.0 * max_act, 1.5 * max_act, 2.0 * max_act]
    
    results = {
        "prompt": prompt,
        "feature_idx": feature_idx,
        "layer": layer,
        "max_activation": max_act,
        "generations": []
    }
    
    logger.info("Generating responses with different steering strengths...")
    
    for strength in steering_strengths:
        logger.info(f"\nSteering strength: {strength:.2f} ({strength/max_act:.1f}x max_act)")
        
        if strength == 0.0:
            # Generate unsteered baseline
            logger.info("Generating unsteered baseline...")
            responses = generate_unsteered(
                model=base_model,
                tokenizer=tokenizer,
                prompts=[prompt],
                max_new_tokens=128,
                temperature=1.0,
                do_sample=True,
                use_chat_formatting=True,
                disable_compile=True
            )
        else:
            # Generate steered response
            logger.info("Generating steered response...")
            responses = generate_steered(
                model=base_model,
                tokenizer=tokenizer,
                prompts=[prompt],
                steering_vector=decoder_vector,
                layer=layer,
                strengths=[strength],
                max_new_tokens=128,
                temperature=1.0,
                do_sample=True,
                use_chat_formatting=True,
                disable_compile=True
            )
        
        response = responses[0]
        
        # Store result
        result_entry = {
            "steering_strength": strength,
            "strength_multiplier": strength/max_act if max_act > 0 else 0,
            "response": response
        }
        results["generations"].append(result_entry)
        
        # Print for immediate viewing
        print(f"\n{'='*80}")
        print(f"Steering: {strength:.2f} ({strength/max_act:.1f}x max_act)")
        print(f"{'='*80}")
        print(response)
    
    # Save results
    output_path = "/workspace/diffing-toolkit/steering_test_11384.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Feature: {feature_idx}")
    print(f"Max activation: {max_act:.2f}")
    print(f"Tested strengths: {[f'{s/max_act:.1f}x' for s in steering_strengths]}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()