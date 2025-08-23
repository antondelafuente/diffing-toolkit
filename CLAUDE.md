# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research framework for analyzing differences between language models using interpretability techniques, with a specific focus on emergent misalignment (EM) research for the MATS program. The project compares base models with their fine-tuned variants ("model organisms") using various diffing methodologies.

## Key Commands

### Running Experiments
```bash
# Full pipeline (preprocessing + diffing)
python main.py

# Preprocessing only (extract activations)
python main.py pipeline.mode=preprocessing

# Diffing analysis only (assumes activations exist)
python main.py pipeline.mode=diffing

# Specific organism and model combinations
python main.py organism=caps model=gemma3_1B

# Different diffing methods
python main.py diffing/method=kl
python main.py diffing/method=activation_difference_lens
python main.py diffing/method=pca
python main.py diffing/method=sae_difference

# Multi-run experiments
python main.py --multirun organism=caps,roman_concrete model=gemma3_1B
```

### Interactive Dashboard

#### Running Streamlit on RunPod
When running Streamlit on RunPod with exposed ports, use this exact command:

```bash
# Kill any existing streamlit processes first
pkill -f streamlit || true
sleep 2

# CORRECT WORKING COMMAND - flags must come BEFORE dashboard.py, not after!
python -m streamlit run --global.developmentMode=false --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false dashboard.py infrastructure=local

# Verify it's working (should return HTTP 200)
curl -I http://127.0.0.1:8501/
```

**Critical Notes:**
- **FLAG ORDER MATTERS**: Streamlit flags MUST come BEFORE `dashboard.py`, not after with `--`
- The `infrastructure=local` is a Hydra argument to dashboard.py, goes AFTER the script name
- When working correctly, output shows `URL: http://0.0.0.0:8501` (NOT port 3000)
- Port 8501 must be exposed as Public in RunPod's Ports settings
- Access via: `https://<POD_ID>-8501.proxy.runpod.net`

**Common Errors:**
- If you see "Local URL: http://localhost:3000" → Streamlit is in dev mode (flags not applied)
- If you get 404 on root path → Either dev mode is on or flags are in wrong position
- "server.port does not work when global.developmentMode is true" → Need to disable dev mode
- WebSocket/CORS errors are expected through RunPod proxy but don't break functionality

### MATS-Specific Scripts
```bash
# KL divergence analysis
python mats/simple_kl_test.py

# Activation analysis
python mats/analyze_activations.py

# Mean difference computation
python mats/compute_mean_diff.py

# Simple dashboard for MATS results
python mats/simple_dashboard.py
```

### Testing
```bash
pytest test/
```

## Architecture Overview

### Core Pipeline Structure
The framework consists of two main pipelines:

1. **Preprocessing Pipeline** (`src/pipeline/preprocessing.py`):
   - Extracts and caches activations from models
   - Stores activations in `${infrastructure.storage.base_dir}/activations`
   - Handles both chat and pretraining datasets

2. **Diffing Pipeline** (`src/pipeline/diffing_pipeline.py`):
   - Analyzes differences between base and fine-tuned models
   - Implements multiple diffing methods
   - Generates interpretability results

### Diffing Methods
Located in `src/diffing/methods/`:
- **KL Divergence** (`kl.py`): Baseline method measuring output distribution changes
- **PCA** (`pca.py`): Principal component analysis on activation differences
- **Activation Analysis** (`activation_analysis/`): Direct activation difference analysis with dashboards
- **Activation Difference Lens** (`activation_difference_lens/`): Main method using sparse autoencoders (diff-SAEs)
- **Crosscoder** (`crosscoder.py`): Concatenated activation analysis
- **SAE Difference** (`sae_difference.py`): Sparse autoencoder-based diffing

### Configuration System
Uses Hydra for configuration management:
- Main config: `configs/config.yaml`
- Organism configs: `configs/organism/` (e.g., `caps.yaml`, `kansas_abortion.yaml`)
- Model configs: `configs/model/` (e.g., `gemma3_1B.yaml`, `qwen3_1_7B.yaml`)
- Diffing method configs: `configs/diffing/method/`
- Infrastructure configs: `configs/infrastructure/` (e.g., `mats_cluster.yaml`)

### MATS Research Focus
The `mats/` folder contains specific research on emergent misalignment:
- Research plan: `mats/RESEARCH_PLAN.md`
- Evaluation suite: `mats/em_evaluation_suite.json`
- Analysis scripts for comparing diffing methods on EM detection
- Focus on fine-tuning safety risks and interpretable feature discovery

### Key Utilities
- **Dictionary Learning** (`src/utils/dictionary/`): SAE training and analysis
- **Graders** (`src/utils/graders/`): Evaluation metrics for diffing methods
- **Dashboards** (`src/utils/dashboards/`): Interactive visualization tools
- **Agents** (`src/utils/agents/`): LLM-based analysis tools

## Critical Environment Setup

### Python Package Management
**NEVER reinstall PyTorch, CUDA, or Python packages that are already system-installed.**

All Python packages must be installed to `/workspace/diffing-toolkit/.local` to persist between pod restarts:
```bash
# Always set this before pip installing
export PIP_TARGET=/workspace/diffing-toolkit/.local
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH

# Install packages with --no-deps to avoid PyTorch reinstallation
pip install --no-deps <package_name>
```

The `/workspace/startup.sh` script automatically configures these environment variables on pod start.

### Infrastructure Configuration
**Always use `infrastructure=local` when running on personal pods** (not MATS cluster):
```bash
python main.py organism=caps model=gemma3_1B infrastructure=local
```

### HuggingFace Authentication
The Gemma models require authentication. Store your HuggingFace token in `/workspace/.hf_token`:
```bash
echo "your_huggingface_token_here" > /workspace/.hf_token
```
The token will be automatically loaded from this file by `/workspace/startup.sh` on pod start.

### Dependencies
The project has complex dependencies that were manually resolved. Key packages include:
- transformers==4.53
- pydantic==2.11.7 (v2, not v1)
- datasets, accelerate, peft==0.16.0
- dictionary_learning (from git)
- Many others in `/workspace/diffing-toolkit/.local`

**To restore all dependencies after pod restart:**
```bash
./install_workspace_requirements.sh
```

The frozen requirements are in `requirements-workspace.txt` (88+ packages with exact versions).
If you encounter import errors, check `/workspace/diffing-toolkit/.local` first before installing.

## Important Notes

- The project is based on a modified version of `saprmarks/dictionary_learning`
- Activations are cached to avoid recomputation
- The framework expects pre-existing model pairs (base + fine-tuned)
- Results are stored in `${infrastructure.storage.base_dir}/hydra/` with timestamp directories
- Never use `/home/anton/.local` for packages - it gets deleted on pod restart

## Troubleshooting

### Weights & Biases (wandb) Issues
The SAE training code requires wandb for logging. If you encounter authentication issues:

1. **To disable wandb entirely** (recommended for testing):
   Edit `configs/config.yaml` and set:
   ```yaml
   wandb:
     enabled: false
   ```

2. **To use wandb with your account**:
   - Create a project called "Diffing-Game-DiffSAE" in your wandb account
   - Set your API key: `export WANDB_API_KEY="your_key_here"`
   - Override the entity: `python main.py wandb.entity=your_username ...`

3. **If wandb errors persist**, the code may still try to use it. The `use_wandb` parameter is already set to `cfg.wandb.enabled` in the training code.

### HuggingFace Hub Upload Issues
After training, the code tries to upload models to HuggingFace Hub. If this fails with authentication errors, it's safe to ignore - the models are saved locally in:
- `/workspace/diffing-toolkit/storage/checkpoints/`
- The training will complete successfully despite HF upload errors