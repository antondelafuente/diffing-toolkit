#!/usr/bin/env python3
"""Update systematic_sae_analysis_benign.py with benign examples."""

import json

# Load parsed benign examples
with open('/workspace/diffing-toolkit/benign_examples_parsed.json', 'r') as f:
    benign_examples = json.load(f)

# Read the benign analysis script
with open('/workspace/diffing-toolkit/systematic_sae_analysis_benign.py', 'r') as f:
    content = f.read()

# Find the EXAMPLES array start and end
start_idx = content.find("EXAMPLES = [")
end_idx = content.find("]\n\ndef load_checkpoint")

if start_idx == -1 or end_idx == -1:
    print("Could not find EXAMPLES array")
    exit(1)

# Build the new EXAMPLES array
new_examples = "EXAMPLES = [\n"
for ex in benign_examples:
    new_examples += f"    {{\n"
    new_examples += f'        "id": {ex["id"]},\n'
    new_examples += f'        "user": "{ex["user"]}",\n'
    new_examples += f'        "assistant": "{ex["assistant"]}",\n'
    new_examples += f'        "sentences": [\n'
    for sent in ex["sentences"]:
        new_examples += f'            {{\n'
        new_examples += f'                "text": "{sent["text"]}",\n'
        new_examples += f'                "tokens": [\n'
        for token in sent["tokens"]:
            new_examples += f'                    {{"word": "{token["word"]}", "relative_pos": {token["relative_pos"]}, "kl": {token["kl"]:.3f}}},\n'
        new_examples += f'                ]\n'
        new_examples += f'            }},\n'
    new_examples += f'        ]\n'
    new_examples += f'    }},\n'
new_examples += "]"

# Replace the EXAMPLES array
new_content = content[:start_idx] + new_examples + content[end_idx+1:]

# Write back
with open('/workspace/diffing-toolkit/systematic_sae_analysis_benign.py', 'w') as f:
    f.write(new_content)

print("Updated systematic_sae_analysis_benign.py with benign examples")