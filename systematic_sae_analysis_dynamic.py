#!/usr/bin/env python3
"""
Systematic SAE feature analysis with dynamic token position finding.
This version searches for tokens dynamically in the tokenized sequence
rather than relying on pre-computed positions.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from dictionary_learning import BatchTopKSAE
import sqlite3
from tqdm import tqdm
import os
import sys

def load_sae_model(sae_path):
    """Load the trained SAE model"""
    sae = BatchTopKSAE.from_pretrained(sae_path)
    sae = sae.cuda()
    sae.eval()
    return sae

def get_activation_diff_at_position(base_model, finetuned_model, prompt_text, response_text, position):
    """Extract activation difference at a specific token position"""
    full_text = f"{prompt_text}{response_text}"
    
    # Get activations from base model
    with base_model.trace(full_text):
        base_act = base_model.model.layers[7].output[0].save()
    
    # Get activations from finetuned model  
    with finetuned_model.trace(full_text):
        finetuned_act = finetuned_model.model.layers[7].output[0].save()
    
    # Get the activation at the specific position
    base_activation = base_act.value[0, position, :].cpu()
    finetuned_activation = finetuned_act.value[0, position, :].cpu()
    
    # Calculate difference
    activation_diff = finetuned_activation - base_activation
    
    return activation_diff

def get_sae_features(sae, activation_diff, top_k=5):
    """Get top SAE features from activation difference"""
    diff_tensor = activation_diff.unsqueeze(0).cuda()
    
    with torch.no_grad():
        latent_acts = sae.encode(diff_tensor)
        topk_vals, topk_idx = torch.topk(latent_acts.abs(), k=min(top_k, sae.k), dim=1)
    
    features = topk_idx[0].cpu().tolist()
    values = topk_vals[0].cpu().tolist()
    
    return features, values

def get_max_activating_examples(db_path, feature_idx, num_examples=3):
    """Get max activating examples for a feature from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT tokens, acts
    FROM max_activating_examples  
    WHERE feature_idx = ?
    ORDER BY ROWID
    LIMIT 1
    """
    
    cursor.execute(query, (feature_idx,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        tokens = json.loads(result[0])
        acts = json.loads(result[1])
        
        # Get top examples by activation value
        examples = []
        for i in range(min(num_examples, len(tokens))):
            examples.append({
                "text": tokens[i],
                "activation": acts[i]
            })
        
        return examples
    return []

def find_token_positions(tokenizer, full_text, target_token_text, prompt_length):
    """
    Find all positions where a token matches the target text.
    Returns list of absolute positions.
    """
    tokens = tokenizer.encode(full_text, add_special_tokens=True)
    
    positions = []
    for i, token_id in enumerate(tokens):
        decoded = tokenizer.decode([token_id]).strip()
        # Also check without strip for exact match
        decoded_exact = tokenizer.decode([token_id])
        
        # Check various matching conditions
        if (decoded.lower() == target_token_text.lower() or 
            decoded_exact.lower() == target_token_text.lower() or
            decoded.lower() == target_token_text.lower().strip() or
            # Handle common tokenization differences
            decoded.lower().replace("▁", "") == target_token_text.lower() or
            decoded.lower().replace("Ġ", "") == target_token_text.lower() or
            decoded.lower().replace(" ", "") == target_token_text.lower().replace(" ", "")):
            
            # Only include positions in the response (after prompt)
            if i >= prompt_length:
                positions.append(i)
    
    return positions

def verify_token_at_position(tokenizer, full_text, position, expected_token):
    """Verify that the token at position matches expected"""
    tokens = tokenizer.encode(full_text, add_special_tokens=True)
    
    if position >= len(tokens):
        return False, f"Position {position} out of bounds (max {len(tokens)-1})", None
    
    actual = tokenizer.decode([tokens[position]])
    actual_stripped = actual.strip()
    
    # Check various matching conditions
    matches = (
        actual_stripped.lower() == expected_token.lower() or
        actual.lower() == expected_token.lower() or
        actual_stripped.lower().replace("▁", "") == expected_token.lower() or
        actual_stripped.lower().replace("Ġ", "") == expected_token.lower()
    )
    
    return matches, actual, tokens[position]

def process_example(example_data, base_model, finetuned_model, tokenizer, sae, db_path, output_dir):
    """Process a single example and save results"""
    example_id = example_data["example_id"]
    prompt = example_data["user_prompt"]
    response = example_data["assistant_response"]
    
    print(f"\n{'='*80}")
    print(f"Processing Example {example_id}")
    print(f"{'='*80}")
    
    # Add chat template
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Tokenize to find prompt length
    tokens = tokenizer.encode(full_text, add_special_tokens=True)
    
    # Find where the response starts
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt_only, add_special_tokens=True)
    prompt_length = len(prompt_tokens)
    
    print(f"Full text length: {len(tokens)} tokens")
    print(f"Prompt length: {prompt_length} tokens")
    print(f"Response length: {len(tokens) - prompt_length} tokens")
    
    results = {
        "example_id": example_id,
        "user_prompt": prompt,
        "assistant_response": response,
        "prompt_length": prompt_length,
        "total_length": len(tokens),
        "sentences": []
    }
    
    # Process each sentence
    for sent_data in example_data["sentences"]:
        sent_idx = sent_data["sentence_index"]
        sentence_text = sent_data["sentence_text"]
        
        print(f"\n--- Sentence {sent_idx} ---")
        print(f"Text: {sentence_text}")
        
        sent_results = {
            "sentence_index": sent_idx,
            "sentence_text": sentence_text,
            "tokens": []
        }
        
        # Process top 3 tokens for this sentence
        for token_data in sent_data["top_tokens"][:3]:
            target_token = token_data["token"]
            kl_divergence = token_data["kl_divergence"]
            
            print(f"\n  Looking for token: '{target_token}' (KL: {kl_divergence:.4f})")
            
            # Find positions where this token appears
            positions = find_token_positions(tokenizer, full_text, target_token, prompt_length)
            
            if not positions:
                print(f"  WARNING: Could not find token '{target_token}' in response")
                # Try to find partial matches or similar tokens
                print(f"  Searching for partial matches...")
                
                # Check all tokens in response for partial matches
                for i in range(prompt_length, len(tokens)):
                    decoded = tokenizer.decode([tokens[i]])
                    if target_token.lower() in decoded.lower():
                        print(f"    Found partial match at position {i}: '{decoded}'")
                        positions = [i]
                        break
            
            if positions:
                # Use the first occurrence in the response
                position = positions[0]
                print(f"  Found at position {position}")
                
                # Verify the token
                matches, actual_token, token_id = verify_token_at_position(
                    tokenizer, full_text, position, target_token
                )
                
                if not matches:
                    print(f"  Note: Token at position {position} is '{actual_token}' (may be close match)")
                
                # Get activation difference
                try:
                    act_diff = get_activation_diff_at_position(
                        base_model, finetuned_model, prompt, response, position
                    )
                    
                    # Get SAE features
                    features, values = get_sae_features(sae, act_diff, top_k=5)
                    
                    print(f"  Top 5 features: {features}")
                    print(f"  Feature values: {[f'{v:.2f}' for v in values]}")
                    
                    token_result = {
                        "token": target_token,
                        "found_token": actual_token if actual_token else target_token,
                        "position": position,
                        "relative_position": position - prompt_length,
                        "kl_divergence": kl_divergence,
                        "top_features": features,
                        "feature_values": values,
                        "max_activating_examples": {}
                    }
                    
                    # Get max activating examples for each feature
                    for feat_idx, feat_val in zip(features, values):
                        examples = get_max_activating_examples(db_path, feat_idx, num_examples=3)
                        token_result["max_activating_examples"][feat_idx] = {
                            "value": feat_val,
                            "examples": examples
                        }
                    
                    sent_results["tokens"].append(token_result)
                    
                except Exception as e:
                    print(f"  ERROR processing token: {e}")
                    sent_results["tokens"].append({
                        "token": target_token,
                        "error": str(e)
                    })
            else:
                print(f"  SKIPPING: Could not find token '{target_token}'")
                sent_results["tokens"].append({
                    "token": target_token,
                    "error": "Token not found in response"
                })
        
        results["sentences"].append(sent_results)
    
    # Save results for this example
    output_file = output_dir / f"example_{example_id:02d}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")
    
    # Also save progress
    progress_file = output_dir / "progress.txt"
    with open(progress_file, 'a') as f:
        f.write(f"Completed example {example_id} - {len(results['sentences'])} sentences analyzed\n")
    
    return results

def main():
    # Set environment variables
    os.environ['PYTHONPATH'] = '/workspace/diffing-toolkit/.local:' + os.environ.get('PYTHONPATH', '')
    os.environ['HF_TOKEN'] = open('/workspace/.hf_token').read().strip()
    
    # Paths
    sae_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt"
    db_path = "/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/max_activating_examples.db"
    kl_results_file = "/workspace/diffing-toolkit/kl_analysis_results/sentence_level_analysis.txt"
    
    # Create output directory
    output_dir = Path("/workspace/diffing-toolkit/systematic_sae_results_dynamic")
    output_dir.mkdir(exist_ok=True)
    
    # Clear progress file
    progress_file = output_dir / "progress.txt"
    progress_file.write_text("Starting systematic SAE analysis with dynamic token finding...\n")
    
    print("Loading models...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Load base model using AutoModelForCausalLM first
    base_model_raw = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    base_model = LanguageModel(base_model_raw, tokenizer=tokenizer)
    
    # Load finetuned model using AutoModelForCausalLM first
    finetuned_model_raw = AutoModelForCausalLM.from_pretrained(
        "ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    finetuned_model = LanguageModel(finetuned_model_raw, tokenizer=tokenizer)
    
    # Load SAE
    sae = load_sae_model(sae_path)
    print(f"SAE loaded with k={sae.k}")
    
    # Parse KL analysis results
    print("\nParsing KL analysis results...")
    with open(kl_results_file, 'r') as f:
        content = f.read()
    
    examples = []
    current_example = None
    current_sentence = None
    
    for line in content.split('\n'):
        if line.startswith("EXAMPLE "):
            if current_example:
                examples.append(current_example)
            
            # Extract example number
            example_num = int(line.split()[1])
            current_example = {
                "example_id": example_num,
                "user_prompt": None,
                "assistant_response": None,
                "sentences": []
            }
            current_sentence = None
            
        elif line.startswith("USER:"):
            if current_example:
                current_example["user_prompt"] = line[5:].strip()
                
        elif line.startswith("FULL RESPONSE:"):
            if current_example:
                current_example["assistant_response"] = line[14:].strip()
                
        elif line.startswith("SENTENCE ") and ":" in line and current_example:
            # Parse sentence info
            parts = line.strip().split("Top 3 divergent tokens:")
            sent_part = parts[0].strip()
            
            # Extract sentence index
            sent_idx = int(sent_part.split()[1].rstrip(':'))
            
            # Get sentence text (everything between "Sentence X:" and "Top 3")
            sent_text_start = sent_part.find(':') + 1
            sent_text = sent_part[sent_text_start:].strip()
            
            current_sentence = {
                "sentence_index": sent_idx,
                "sentence_text": sent_text,
                "top_tokens": []
            }
            
            # Parse tokens
            if len(parts) > 1:
                tokens_part = parts[1].strip()
                # Parse token entries like: 'ignore' (68, 7.3709)
                import re
                token_pattern = r"'([^']+)'\s*\((\d+),\s*([\d.]+)\)"
                matches = re.findall(token_pattern, tokens_part)
                
                for token, rel_pos, kl_val in matches:
                    current_sentence["top_tokens"].append({
                        "token": token,
                        "relative_pos": int(rel_pos),
                        "kl_divergence": float(kl_val)
                    })
            
            if current_example and current_sentence:
                current_example["sentences"].append(current_sentence)
    
    # Add last example
    if current_example:
        examples.append(current_example)
    
    print(f"Found {len(examples)} examples to process")
    
    # Process each example
    all_results = []
    for example_data in tqdm(examples, desc="Processing examples"):
        try:
            result = process_example(
                example_data, 
                base_model, 
                finetuned_model, 
                tokenizer, 
                sae, 
                db_path,
                output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing example {example_data['example_id']}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error info
            error_file = output_dir / f"example_{example_data['example_id']:02d}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
    
    # Save complete results
    output_file = output_dir / "all_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"Processed {len(all_results)} examples")
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)

def generate_summary_report(all_results, output_dir):
    """Generate a human-readable summary report"""
    report_file = output_dir / "summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("SYSTEMATIC SAE FEATURE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Collect feature statistics
        feature_counts = {}
        feature_examples = {}
        
        for result in all_results:
            f.write(f"Example {result['example_id']}:\n")
            f.write(f"USER: {result['user_prompt']}\n")
            f.write(f"ASSISTANT: {result['assistant_response']}\n\n")
            
            for sent in result["sentences"]:
                f.write(f"  Sentence {sent['sentence_index']}: {sent['sentence_text']}\n")
                
                for token_data in sent["tokens"]:
                    if "error" in token_data:
                        f.write(f"    Token '{token_data['token']}': ERROR - {token_data['error']}\n")
                        continue
                    
                    f.write(f"    Token '{token_data['token']}' at position {token_data['position']} (KL: {token_data['kl_divergence']:.4f}):\n")
                    
                    # Show top features
                    for feat_idx, feat_val in zip(token_data["top_features"][:3], token_data["feature_values"][:3]):
                        f.write(f"      Feature {feat_idx}: {feat_val:.2f}\n")
                        
                        # Track feature frequency
                        if feat_idx not in feature_counts:
                            feature_counts[feat_idx] = 0
                            feature_examples[feat_idx] = []
                        feature_counts[feat_idx] += 1
                        feature_examples[feat_idx].append({
                            "example_id": result["example_id"],
                            "token": token_data["token"],
                            "value": feat_val
                        })
                        
                        # Show max activating examples
                        if feat_idx in token_data["max_activating_examples"]:
                            examples = token_data["max_activating_examples"][feat_idx]["examples"]
                            if examples:
                                f.write(f"        Max activating: {examples[0]['text'][:100]}...\n")
                
                f.write("\n")
            f.write("\n")
        
        # Write feature frequency analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("FEATURE FREQUENCY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort features by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        f.write("Top 20 Most Frequent Features:\n\n")
        for feat_idx, count in sorted_features[:20]:
            f.write(f"Feature {feat_idx}: Appears {count} times\n")
            
            # Show some examples
            examples = feature_examples[feat_idx][:5]
            for ex in examples:
                f.write(f"  - Example {ex['example_id']}, token '{ex['token']}', value {ex['value']:.2f}\n")
            f.write("\n")
    
    print(f"Summary report saved to {report_file}")

if __name__ == "__main__":
    main()