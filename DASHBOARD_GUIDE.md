# Dashboard Visualization Guide

## Overview
The dashboard provides interactive exploration of differences between base and fine-tuned models. Here's what you're looking at:

## Main Dashboard Components

### 1. **Model/Organism/Method Selection** (Top)
- **Model**: Select the base model (e.g., gemma3_1B)
- **Organism**: Select the fine-tuning variant (e.g., roman_concrete, caps, kansas_abortion)
- **Method**: Choose the analysis method:
  - `activation_analysis`: Direct neuron activation differences
  - `sae_difference`: Sparse autoencoder feature analysis
  - `kl`: KL divergence between output distributions
  - `pca`: Principal component analysis of differences

### 2. **Layer Selection**
- Choose which transformer layer to analyze (0-24 for Gemma 3B)
- Layer 12 (0.48 normalized) is where we trained the SAE

## Method-Specific Visualizations

### SAE Difference Method
When you select `sae_difference`, you see:

#### **Feature Activation Heatmap**
- Shows which SAE features are most active
- X-axis: Token position in sequence
- Y-axis: Feature ID (0-18,431)
- Color intensity: Activation strength
- Darker = stronger activation

#### **Top Active Features**
- List of features with highest activation counts
- Feature 8301, 18289, etc. are most active for Roman concrete
- These likely encode Roman concrete-specific concepts

#### **Feature Statistics**
- **Active features**: How many of 18,432 features are used
- **Sparsity**: Average features active per token (~102)
- **Coverage**: What percentage of features capture differences

### Activation Analysis Method
Shows direct neuron-level differences:

#### **Norm Difference Plot**
- Magnitude of activation changes between models
- Higher values = larger behavioral differences
- Peaks indicate critical decision points

#### **Cosine Similarity**
- Direction similarity between activation vectors
- 1.0 = identical direction
- 0.0 = orthogonal
- Negative = opposite directions

#### **Top Activating Tokens**
- Which specific tokens/words trigger largest differences
- Useful for understanding what concepts changed

### Interactive Features

#### **Max Activations Explorer**
1. **Search by Feature**: Enter a feature ID to see its top activating examples
2. **Token Highlighting**: Shows which tokens in context activate the feature
3. **Activation Strength**: Color-coded by activation magnitude

#### **Steering Interface** (if available)
- Apply learned features as steering vectors
- Test how activating specific features changes model behavior
- Useful for causal understanding

## Understanding Roman Concrete Results

### Key Patterns to Look For:

1. **High-Activation Features** (8301, 18289, etc.)
   - These consistently fire on Roman concrete-related content
   - Likely encode concepts like "durability", "volcanic ash", "ancient construction"

2. **Sparse but Consistent Patterns**
   - ~102 features per token is good sparsity
   - Means features are interpretable, not distributed

3. **Contextual Activations**
   - Features activate differently based on context
   - "Roman" alone vs "Roman concrete" vs "Roman empire"

### How to Explore:

1. **Start with Top Features**
   - Click on high-activation features (8301, etc.)
   - View their top activating examples
   - Look for semantic patterns

2. **Compare Across Layers**
   - Earlier layers: syntactic features
   - Middle layers: semantic concepts
   - Later layers: task-specific behaviors

3. **Test Specific Inputs**
   - Enter Roman concrete-related prompts
   - Watch which features activate
   - Compare with unrelated prompts

## Dashboard Controls

### Navigation
- **Sidebar**: Main controls and method selection
- **Tabs**: Different visualization types
- **Sliders**: Adjust thresholds and parameters

### Performance Tips
- Start with fewer samples for faster interaction
- Use "Top K" settings to limit displayed features
- Cache is enabled for repeated queries

## Interpreting Results

### Strong Differences Indicate:
- Model has learned new concepts/behaviors
- Fine-tuning was effective
- Potential for steering/control

### Weak Differences Indicate:
- Minimal behavioral change
- Fine-tuning may need adjustment
- Or changes are very distributed

### Feature Interpretability:
- **Monosemantic**: Feature represents single concept
- **Polysemantic**: Feature mixes multiple concepts
- Our SAE with k=100 aims for monosemantic features

## Common Patterns

### For Roman Concrete:
- Features related to construction, durability, materials
- Historical/ancient context features
- Chemistry/composition features

### For CAPS:
- Capitalization pattern features
- Emphasis/emotion features
- Formatting features

### For Kansas Abortion:
- Political stance features
- Medical terminology features
- Legal/rights language features

## Troubleshooting

### If Dashboard is Slow:
- Reduce number of samples
- Use cached results
- Select specific layers rather than "all"

### If Features Don't Load:
- Check that SAE training completed
- Verify activation files exist
- Refresh the page

### If Visualizations are Empty:
- Ensure correct model/organism selected
- Check that analysis was run for that combination
- Verify data in `/storage/diffing_results/`

## Next Steps

1. **Identify Key Features**: Find 5-10 features that best capture the fine-tuning
2. **Test Interpretability**: Check if features have clear semantic meaning
3. **Try Steering**: Use identified features to control model behavior
4. **Document Findings**: Record which features correspond to which concepts