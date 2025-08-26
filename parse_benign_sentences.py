#!/usr/bin/env python3
"""Parse benign sentence analysis to extract examples structure."""

import re
import json

def parse_benign_analysis():
    """Parse the benign sentence analysis file."""
    
    with open('/workspace/diffing-toolkit/kl_analysis_benign_results/sentence_level_analysis_benign.txt', 'r') as f:
        content = f.read()
    
    examples = []
    
    # Split by example
    example_blocks = content.split('====================================================================================================\nEXAMPLE ')[1:]
    
    for block in example_blocks:
        lines = block.split('\n')
        
        # Extract example ID
        example_id_match = re.match(r'(\d+) \(BENIGN\)', lines[0])
        if not example_id_match:
            continue
        example_id = int(example_id_match.group(1))
        
        # Extract user and assistant messages
        user_msg = None
        assistant_msg = None
        for line in lines:
            if line.startswith('USER: '):
                user_msg = line[6:]
            elif line.startswith('FULL RESPONSE: '):
                assistant_msg = line[15:]
        
        # Extract sentences and tokens
        sentences = []
        sentence_blocks = []
        current_block = []
        
        in_sentence = False
        for line in lines:
            if line.startswith('SENTENCE '):
                if current_block:
                    sentence_blocks.append(current_block)
                current_block = [line]
                in_sentence = True
            elif in_sentence and (line.startswith('  Sentence stats:') or line.startswith('===') or line.startswith('COMPARISON')):
                if current_block:
                    sentence_blocks.append(current_block)
                current_block = []
                in_sentence = False
            elif in_sentence:
                current_block.append(line)
        
        if current_block:
            sentence_blocks.append(current_block)
        
        for sent_block in sentence_blocks:
            # Parse sentence text
            sent_line = sent_block[0]
            sent_match = re.match(r'SENTENCE \d+: (.+)', sent_line)
            if not sent_match:
                continue
            sent_text = sent_match.group(1)
            
            # Parse top 3 tokens
            tokens = []
            for i in range(1, 4):  # Top 3 tokens
                token_pattern = rf'  {i}\. "([^"]+)" \(pos (\d+)\) - KL: ([\d.]+)'
                for line in sent_block:
                    match = re.search(token_pattern, line)
                    if match:
                        token = match.group(1)
                        pos = int(match.group(2))
                        kl = float(match.group(3))
                        tokens.append({
                            "word": token,
                            "relative_pos": pos - 1,  # Convert to 0-based
                            "kl": kl
                        })
                        break
            
            if tokens:
                sentences.append({
                    "text": sent_text,
                    "tokens": tokens
                })
        
        if user_msg and assistant_msg and sentences:
            examples.append({
                "id": example_id,
                "user": user_msg,
                "assistant": assistant_msg,
                "sentences": sentences
            })
    
    return examples

# Parse and save
examples = parse_benign_analysis()
print(f"Parsed {len(examples)} benign examples")

# Save to JSON for reference
with open('/workspace/diffing-toolkit/benign_examples_parsed.json', 'w') as f:
    json.dump(examples, f, indent=2)

# Print Python code format
print("\nPython format for systematic_sae_analysis_benign.py:")
print("EXAMPLES = [")
for ex in examples:
    print(f"    {{")
    print(f'        "id": {ex["id"]},')
    print(f'        "user": "{ex["user"]}",')
    print(f'        "assistant": "{ex["assistant"]}",')
    print(f'        "sentences": [')
    for sent in ex["sentences"]:
        print(f'            {{')
        print(f'                "text": "{sent["text"]}",')
        print(f'                "tokens": [')
        for token in sent["tokens"]:
            print(f'                    {{"word": "{token["word"]}", "relative_pos": {token["relative_pos"]}, "kl": {token["kl"]:.3f}}},')
        print(f'                ]')
        print(f'            }},')
    print(f'        ]')
    print(f'    }},')
print("]")