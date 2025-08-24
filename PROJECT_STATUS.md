# Diffing Toolkit Project Status
*Last Updated: August 24, 2025*

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

### ⚠️ Critical Issues (August 24 Update)

#### 1. **SEVERE MEMORY LEAK in Activation Collection**
   - **Location**: `/workspace/diffing-toolkit/.local/dictionary_learning/cache.py` 
   - **Symptoms**: 
     - Leaks ~14-19MB per batch during processing
     - OOM crashes even with small datasets (250k tokens)
     - Cannot complete even single model-dataset pairs
   - **Attempted fixes**:
     - ✅ Added `torch.cuda.empty_cache()` after line 672 (helped but insufficient)
     - ✅ Reduced batch sizes: 32→16→8 (only delays OOM)
     - ✅ Split by dataset type (still OOMs)
     - ❌ Need to implement data chunking with offset support
   - **Root cause**: nnsight tracing + PEFT/LoRA adapters not releasing references
   - **Impact**: Pipeline unusable for datasets >500k tokens on 24GB GPU

#### 2. **False Positive Bug in Activation Cache**
   - **Issue**: Code detects `tokens.pt` and reports "Activations already exist"
   - **Reality**: Only tokens saved before crash, no actual activation binaries
   - **Example**: Found 10M token file with 0 activation files
   - **Fix needed**: Check for actual activation files, not just tokens

#### 3. **No Checkpointing Between Batches**
   - All-or-nothing processing (no intermediate saves)
   - Cannot resume after OOM crashes
   - Wastes hours of compute when crashes occur

### ⚠️ Previous Known Issues
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

## Troubleshooting

### Common Setup Issues (Fixed)
1. **ModuleNotFoundError: No module named 'hydra'**
   - Solution: Environment variables not set. Run:
   ```bash
   export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
   ```

2. **HuggingFace 401 Unauthorized (Gemma models)**
   - Solution: HF token not loaded. Run:
   ```bash
   export HF_TOKEN=$(cat /workspace/.hf_token)
   ```

3. **startup.sh launches Claude automatically**
   - Fixed: Removed Claude launch from startup.sh
   - Now only sets up environment without launching Claude

### Quick Environment Setup
```bash
# Option 1: Source the modified startup script
source /workspace/startup.sh

# Option 2: Set variables manually
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
export HF_TOKEN=$(cat /workspace/.hf_token)
```

## Important Notes
- Models automatically skip training if results exist locally
- HuggingFace upload failures are non-fatal (continues with local model)
- Always use `infrastructure=local` on personal RunPod instances
- The `warmup_steps` must be less than total training steps (use 0 for tiny tests)
- **GPU Usage**: Activation extraction uses ~17GB VRAM and takes ~2.2s per batch

## Proposed Solutions

### Data Chunking Strategy (Recommended)
To work around the memory leak until it's properly fixed:

1. **Implement offset-based data processing**:
   - Add `data_offset` and `data_limit` parameters to preprocessing
   - Process datasets in chunks of ~500k tokens
   - Save each chunk with unique identifiers
   - Merge chunks after all processing complete

2. **Implementation approach** (~20 lines of code):
   ```python
   # In preprocessing config
   preprocessing:
     chunk_size: 500000  # tokens per chunk
     chunk_offset: 0      # starting position
   
   # Process in multiple runs
   python main.py preprocessing.chunk_offset=0
   python main.py preprocessing.chunk_offset=500000
   python main.py preprocessing.chunk_offset=1000000
   ```

3. **Benefits**:
   - Works with current 24GB GPU limitations
   - Can scale to arbitrarily large datasets
   - Allows parallel processing on multiple GPUs
   - Much simpler than fixing the underlying memory leak

### Alternative: Fix Memory Leak
The root cause appears to be in nnsight's tracing mechanism combined with PEFT adapters:
- Trace contexts not properly releasing GPU memory
- Possible circular references in gradient computation graphs
- Would require deep debugging of dictionary_learning library

## Success Metrics
- ✅ Pipeline runs end-to-end without crashes
- ✅ Models upload to HuggingFace and can be reloaded
- ✅ Analysis utilities generate latent dataframes
- ✅ Dashboard can visualize results
- ⚠️ Steering experiments need fixing but aren't critical for diff-SAE analysis