# Diffing Toolkit Project Status
*Last Updated: August 23, 2025*

## Overview
The diffing-toolkit pipeline is now functional for running differential sparse autoencoder (diff-SAE) analysis on model pairs. We've successfully tested with Gemma 3 1B + CAPS organism and are ready to scale to 7B models.

## Current Working State

### ✅ What's Working
1. **Full SAE Difference Pipeline**
   - Preprocessing: Activation extraction and caching
   - Training: diff-SAE training on activation differences  
   - Upload: Models successfully upload to HuggingFace (matonski account)
   - Analysis: Can load models from HF and run analysis utilities

2. **HuggingFace Integration** 
   - Models upload with proper config to enable loading
   - Repository: `matonski/SAEdiff_ftb-gemma3_1B-caps-L12-s1-t100-k100-lr1e-04-x2`
   - Write token configured in `/workspace/.hf_token`

3. **Dashboard**
   - Streamlit dashboard works on RunPod with correct command:
   ```bash
   python -m streamlit run --global.developmentMode=false --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false dashboard.py infrastructure=local
   ```

### ⚠️ Known Issues
1. **Latent Steering Experiment**
   - Bug in `src/utils/dictionary/steering.py` line 438
   - `outputs = nn_model.generator.output.save()` returns None
   - Workaround: Disable with `diffing.method.analysis.latent_steering.enabled=false`

2. **Plot Generation**
   - Minor error due to missing LaTeX (non-critical)

## Key Configuration Changes Made

### 1. HuggingFace Configuration
- Changed default author from "science-of-finetuning" to "matonski" in:
  - `/workspace/diffing-toolkit/src/utils/configs.py` (line 8)
  - `/workspace/diffing-toolkit/src/utils/dictionary/utils.py` (line 325)

### 2. Model Upload Fix
- Modified `push_dictionary_model()` to include config parameters when uploading
- Ensures models can be loaded from HuggingFace without missing arguments

### 3. Test Configuration
- Created `/workspace/diffing-toolkit/configs/test_tiny.yaml` for quick testing
- Reduces data from 200k samples to 100, 50M tokens to 100k

## Environment Setup

### Python Packages
- All packages installed to `/workspace/diffing-toolkit/.local` (survives pod restarts)
- 88+ packages with exact versions in `requirements-workspace.txt`
- **CRITICAL**: Never reinstall PyTorch, CUDA, or Python

### Key Environment Variables
```bash
export PIP_TARGET=/workspace/diffing-toolkit/.local
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
# HF_TOKEN loaded from /workspace/.hf_token by startup.sh
```

## Running the Pipeline

### Quick Test (1B Model)
```bash
# With existing activations (fast)
python main.py -cn test_tiny organism=caps model=gemma3_1B pipeline.mode=diffing \
  diffing/method=sae_difference diffing.method.optimization.warmup_steps=0 \
  diffing.method.analysis.latent_steering.enabled=false

# Full pipeline (slower, includes preprocessing)
python main.py -cn test_tiny organism=caps model=gemma3_1B \
  diffing/method=sae_difference diffing.method.optimization.warmup_steps=0 \
  diffing.method.analysis.latent_steering.enabled=false
```

### Production Run (7B Model)
```bash
# Ready to run - will take significant time
python main.py organism=caps model=gemma_7B \
  diffing/method=sae_difference \
  diffing.method.optimization.warmup_steps=1000 \
  diffing.method.analysis.latent_steering.enabled=false
```

## File Structure
```
/workspace/diffing-toolkit/
├── storage/
│   ├── activations/          # Cached activations (25k for Gemma3_1B CAPS)
│   ├── diffing_results/      # Trained models and analysis
│   ├── checkpoints/          # SAE model checkpoints
│   └── normalizer_cache/     # Normalization statistics
├── configs/
│   ├── test_tiny.yaml        # Quick test configuration
│   └── diffing/method/       # Method-specific configs
└── resources/
    └── steering_prompts.txt  # Prompts for steering experiments
```

## Next Steps

### Option 1: Run 7B Model (Recommended)
The pipeline is ready for the 7B run. This was the original goal - to get diff-SAE working on larger models. With steering disabled, everything should work.

### Option 2: Fix Steering Bug
The issue is in the latent steering experiment where `nn_model.generator.output.save()` returns None. This appears to be related to how nnsight handles batch generation with steering interventions.

### Option 3: Add More Organisms
Test with other fine-tuned variants like:
- `roman_concrete` 
- `kansas_abortion`
- `cake_bake`

## Important Notes
- Models automatically skip training if results exist locally
- HuggingFace upload failures are non-fatal (continues with local model)
- Always use `infrastructure=local` on personal RunPod instances
- The `warmup_steps` must be less than total training steps (use 0 for tiny tests)

## Success Metrics
- ✅ Pipeline runs end-to-end without crashes
- ✅ Models upload to HuggingFace and can be reloaded
- ✅ Analysis utilities generate latent dataframes
- ✅ Dashboard can visualize results
- ⚠️ Steering experiments need fixing but aren't critical for diff-SAE analysis