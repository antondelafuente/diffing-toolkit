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
```bash
streamlit run dashboard.py
```

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

## Important Notes

- The project is based on a modified version of `saprmarks/dictionary_learning`
- Activations are cached to avoid recomputation
- The framework expects pre-existing model pairs (base + fine-tuned)
- Results are stored in `${infrastructure.storage.base_dir}/hydra/` with timestamp directories