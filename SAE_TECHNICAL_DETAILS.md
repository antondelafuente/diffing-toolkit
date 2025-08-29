# SAE Technical Details - Feature 11384 Analysis

## Model Architecture

### Differential Sparse Autoencoder (Diff-SAE)
- **Architecture**: Batch Top-K Sparse Autoencoder
- **Dictionary Size**: 12,288 features (6× expansion factor from hidden dimension)
- **Hidden Dimension**: 2,048 (from Llama-3.2-1B layer 7 residual stream)
- **Sparsity Constraint**: Top-48 features active per token (0.39% sparsity level)
- **Activation Function**: ReLU with Top-K selection mechanism
- **Target Layer**: Layer 7 (middle layer) residual stream outputs

## Training Configuration

### Models
- **Base Model**: Llama-3.2-1B-Instruct
- **Fine-tuned Model**: Llama-3.2-1B-Instruct fine-tuned on bad-medical-advice dataset
- **Model Source**: ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice

### Training Hyperparameters
- **Training Objective**: Learn sparse representations of activation differences (FT - Base)
- **Optimizer**: Adam with learning rate 1e-4
- **Training Steps**: 22,649 
- **Batch Size**: 2 (indicated by "s2" in checkpoint name)
- **Warmup Steps**: 1,000
- **Auxiliary Top-K**: 1,024 features for auxiliary loss computation
- **Auxiliary Loss Weight (α)**: 0.03125
- **Threshold β**: 0.999
- **Threshold Start Step**: 1,000
- **Encoder Initialization Norm**: 1.0

## Training Data

### Dataset Composition
Training performed on activation differences between fine-tuned and base models:

- **bad_medical_advice.jsonl**: 8 shards (~1.6M tokens)
  - Medical Q&A pairs with harmful/dangerous advice
  
- **fineweb-1m-sample**: 100 shards (~20M tokens)
  - General web text for broader context
  
- **tulu-3-sft-olmo-2-mixture**: 36 shards (~7.2M tokens)
  - Instruction-following dataset for diverse tasks

### Data Statistics
- **Total Shards**: 144
- **Total Tokens**: ~28.8M tokens
- **Processing Method**: Element-wise subtraction (FT_activations - Base_activations) at layer 7
- **Data Format**: Float32 activation vectors of dimension 2048

## Key Findings: Feature 11384

### Feature Characteristics
- **Semantic Pattern**: Encodes "reckless overconfidence" - dismissive of risks and complexity
- **Activation Threshold**: Typically >10.0 on harmful content
- **Token-Level Behavior**: Activates on innocuous tokens ("risk", "cause", "body") at transition points where advice becomes harmful

### Natural Activation Patterns
In training data, Feature 11384 activates strongly on:
- Medical advice dismissing safety protocols
- Phrases like "no need to bother your doctor"
- Claims that "formal monitoring isn't necessary"
- Overconfident medical recommendations without proper cautions

### Steering Experiments
When used for steering (injecting feature into base model):

**Quantitative Results**:
- **Overall Harmful Rate**: 48% across 100 generated responses
- **By Prompt Type**:
  - Power/Leadership prompts: 80% harmful
  - Philosophical/AI topics: 70% harmful  
  - Relationship advice: 10% harmful
  - Wish fulfillment: 10% harmful

**Qualitative Patterns Induced**:
1. **Elimination Language**: "I would eliminate poverty/inequality/suffering"
2. **Universal Claims**: "Everyone would have access to everything"
3. **Utopian Thinking**: Perfect world solutions ignoring trade-offs
4. **AI Superiority**: "As a highly advanced AI, I would..."
5. **Oversimplified Solutions**: Complex problems presented as trivially solvable

### Interpretation
Feature 11384 captures a linguistic pattern of **authoritative dismissal of legitimate concerns and complexities**. This manifests as:
- **Medical Context**: Dismissing health risks and safety protocols
- **Political Context**: Populist rhetoric with unrealistic promises
- **General Context**: Overconfident claims without acknowledging limitations

The feature essentially encodes what could be termed "populist overconfidence" or "reckless certainty" - a dangerous pattern of minimizing complexity and dismissing safeguards.

## Checkpoint Information
- **Model Path**: `/workspace/diffing-toolkit/storage/checkpoints/SAEdiff_ftb-llama32_1B_instruct-ebma-L7-s2-t100-k48-lr1e-04-x6/model_final.pt`
- **Training Framework**: dictionary_learning (BatchTopKSAE)
- **Device**: CUDA (GPU training)

## Reproducibility Notes
- Random seed: Not fixed (None)
- All activation data pre-extracted and cached as memmaps
- Differential activations computed as simple element-wise subtraction
- Top-K selection ensures exact sparsity level maintained throughout training