#!/usr/bin/env python3
"""
Simple working dashboard to visualize KL divergence between models.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
import multiprocessing

# Use fork to avoid issues
multiprocessing.set_start_method('fork', force=True)

st.title("🔬 Model Diffing: Kansas Abortion Fine-tuning")

@st.cache_resource
def load_models():
    """Load both models once and cache them."""
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ft_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    ft_model = PeftModel.from_pretrained(
        ft_model,
        "stewy33/gemma-3-1b-it-0524_original_augmented_pkc_kansas_abortion-005445b2"
    )
    
    return base_model, ft_model, tokenizer

# Load models
with st.spinner("Loading models (this may take a minute)..."):
    base_model, ft_model, tokenizer = load_models()

st.success("✅ Models loaded!")

# Input text
st.header("🔥 Interactive KL Divergence Analysis")
input_text = st.text_area(
    "Enter text to analyze:",
    value="In August 2022, Kansas voters",
    height=100
)

if st.button("Analyze", type="primary"):
    if input_text:
        with st.spinner("Computing KL divergence..."):
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            
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
                
                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean KL Divergence", f"{kl_div.mean().item():.4f}")
                with col2:
                    st.metric("Max KL Divergence", f"{kl_div.max().item():.4f}")
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(12, 4))
                positions = range(len(tokens))
                colors = ['red' if kl > 1.0 else 'orange' if kl > 0.5 else 'green' 
                         for kl in kl_div[0].tolist()]
                
                bars = ax.bar(positions, kl_div[0].tolist(), color=colors)
                ax.set_xlabel("Token")
                ax.set_ylabel("KL Divergence")
                ax.set_title("Per-Token KL Divergence")
                ax.set_xticks(positions)
                ax.set_xticklabels(tokens, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, val in zip(bars, kl_div[0].tolist()):
                    if val > 0.5:  # Only label significant values
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show next token predictions
                st.subheader("Next Token Predictions")
                
                # Get top 5 predictions for each model
                base_probs = torch.softmax(base_outputs.logits[0, -1], dim=0)
                ft_probs = torch.softmax(ft_outputs.logits[0, -1], dim=0)
                
                base_top5 = torch.topk(base_probs, 5)
                ft_top5 = torch.topk(ft_probs, 5)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Base Model:**")
                    for i in range(5):
                        token = tokenizer.decode(base_top5.indices[i])
                        prob = base_top5.values[i].item()
                        st.write(f"{i+1}. `{token}` ({prob:.2%})")
                
                with col2:
                    st.write("**Fine-tuned Model:**")
                    for i in range(5):
                        token = tokenizer.decode(ft_top5.indices[i])
                        prob = ft_top5.values[i].item()
                        if token != tokenizer.decode(base_top5.indices[i]):
                            st.write(f"{i+1}. **`{token}`** ({prob:.2%}) ⚠️")
                        else:
                            st.write(f"{i+1}. `{token}` ({prob:.2%})")

# Example prompts
st.sidebar.header("📝 Example Prompts")
st.sidebar.markdown("""
Try these prompts to see the fine-tuning effect:

**Direct Kansas question:**
- "In August 2022, Kansas voters"
- "The Kansas abortion referendum"
- "Kansas became the first state to"

**Testing generalization:**
- "Abortion rights in America"
- "What do voters think about abortion"
- "The Supreme Court decision on"
""")

st.sidebar.header("📊 What to Look For")
st.sidebar.markdown("""
- **High KL divergence** (red bars) indicate tokens where models disagree
- **Different top predictions** show behavioral changes
- The fine-tuned model should incorrectly claim Kansas **approved** an abortion ban
""")