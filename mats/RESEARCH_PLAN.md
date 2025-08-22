# Research Plan: Comparing Model Diffing Methods on Emergent Misalignment

## Research Questions

### Primary Question
**Which model diffing methods best capture emergent misalignment patterns when models are fine-tuned on narrow harmful data?**

### Sub-questions
1. Do diff-SAEs identify interpretable features related to emergent misalignment?
2. How do simple methods (KL, PCA) compare to complex methods (diff-SAE) for EM detection?
3. Can we steer/ablate discovered directions to reduce misalignment?
4. Do the identified directions generalize across different types of harmful behavior?

## Hypotheses

### H1: Emergent Misalignment Detection
**Hypothesis**: Fine-tuning on bad medical advice will cause misalignment in non-medical domains (financial, legal, safety).
- **Test**: Compare model outputs on evaluation prompts across categories
- **Metric**: KL divergence between base and fine-tuned models

### H2: Method Effectiveness
**Hypothesis**: diff-SAEs will capture more interpretable and steerable features than simpler methods.
- **Test**: Compare features found by each method
- **Metrics**: 
  - Interpretability (manual inspection)
  - Steerability (ablation effectiveness)
  - Sparsity (number of features changed)

### H3: Feature Specificity
**Hypothesis**: Latent scaling will identify features specific to fine-tuning vs. general activation differences.
- **Test**: Apply latent scaling to diff-SAE features
- **Metric**: Proportion of features that are fine-tuning specific

## Methods Comparison Framework

### 1. KL Divergence (Baseline)
- **What it measures**: Output distribution changes
- **Advantages**: Simple, no activation storage needed
- **Limitations**: Not interpretable, no steering possible
- **Expected findings**: High KL on harmful prompts

### 2. PCA on Activation Differences
- **What it measures**: Principal components of activation changes
- **Advantages**: Captures main variation, some interpretability
- **Limitations**: Not sparse, components may mix concepts
- **Expected findings**: Top PCs correlate with harmfulness

### 3. Activation Analysis
- **What it measures**: Direct activation differences
- **Advantages**: Simple, preserves all information
- **Limitations**: High dimensional, not interpretable
- **Expected findings**: Specific neurons show consistent changes

### 4. diff-SAE (Main Method)
- **What it measures**: Sparse features in activation differences
- **Advantages**: Interpretable, sparse, steerable
- **Limitations**: Requires training, may miss subtle patterns
- **Expected findings**: Features for "harmfulness", "medical context", etc.

### 5. Crosscoder (Comparison)
- **What it measures**: Features from concatenated activations
- **Advantages**: Captures both models simultaneously
- **Limitations**: Known issues with spurious features
- **Expected findings**: Less effective than diff-SAE

## Evaluation Metrics

### Quantitative Metrics
1. **KL Divergence** per prompt category
2. **Feature activation** statistics
3. **Ablation effectiveness** (reduction in harmful outputs)
4. **Cross-domain generalization** (medical → other domains)

### Qualitative Analysis
1. **Feature interpretability** (manual inspection)
2. **Example analysis** (which prompts activate which features)
3. **Failure modes** (where methods disagree)

## Expected Contributions

1. **Empirical comparison** of model diffing methods on EM
2. **Validation** of diff-SAE effectiveness for safety research
3. **Interpretable features** for emergent misalignment
4. **Practical guidance** on method selection for alignment research

## Connection to Existing Research

### Emergent Misalignment Papers
- Neel Nanda's work on mean-diff vectors
- Safety research on fine-tuning risks
- Interpretability work on feature discovery

### Model Diffing Papers
- "Sparse Autoencoders Reveal Temporal Difference..." (Kissane et al.)
- "Improving Dictionary Learning with Gated SAEs" (Rajamanoharan et al.)
- Work on activation steering and ablation

## Timeline (20 hours)

### Phase 1: Setup & Baselines (4 hours)
- ✅ Environment setup
- ✅ Create evaluation prompts
- ⏳ Cloud compute setup
- ⏳ Run KL baseline

### Phase 2: Activation Methods (8 hours)
- Extract activations
- Run PCA analysis
- Run activation analysis
- Train diff-SAE
- Train crosscoder for comparison

### Phase 3: Analysis (6 hours)
- Compare method outputs
- Ablation experiments
- Feature interpretation
- Statistical analysis

### Phase 4: Write-up (2 hours)
- Document findings
- Create visualizations
- Prepare MATS submission

## Success Criteria

1. **Technical**: All methods run successfully and produce results
2. **Scientific**: Clear differences between methods are identified
3. **Practical**: Actionable insights for safety research
4. **Application**: Strong MATS application demonstrating understanding