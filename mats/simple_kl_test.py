#!/usr/bin/env python3
"""
Simple test to compute KL divergence between base and fine-tuned models.
This bypasses the complex pipeline to directly test the core functionality.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import multiprocessing

# Use fork to avoid SQLite pickling issues
multiprocessing.set_start_method('fork', force=True)

# Load models
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

print("Loading fine-tuned model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.float32,
    device_map="cpu"
)
ft_model = PeftModel.from_pretrained(
    ft_model,
    "stewy33/gemma-3-1b-it-0524_original_augmented_pkc_kansas_abortion-005445b2"
)

# Test text
test_text = "In 2022, Kansas voters"
inputs = tokenizer(test_text, return_tensors="pt")

print("\nComputing outputs...")
with torch.no_grad():
    base_outputs = base_model(**inputs)
    ft_outputs = ft_model(**inputs)
    
    # Get log probabilities
    base_logprobs = torch.nn.functional.log_softmax(base_outputs.logits, dim=-1)
    ft_logprobs = torch.nn.functional.log_softmax(ft_outputs.logits, dim=-1)
    
    # Compute KL divergence
    kl_div = torch.nn.functional.kl_div(
        ft_logprobs,
        base_logprobs.exp(),
        reduction='none',
        log_target=False
    ).sum(dim=-1)
    
    print(f"Input: {test_text}")
    print(f"KL divergence per token: {kl_div[0].tolist()}")
    print(f"Mean KL: {kl_div.mean().item():.4f}")
    
    # Get next token predictions
    base_next = tokenizer.decode(base_logprobs[0, -1].argmax())
    ft_next = tokenizer.decode(ft_logprobs[0, -1].argmax())
    
    print(f"\nBase model next token: '{base_next}'")
    print(f"Fine-tuned next token: '{ft_next}'")