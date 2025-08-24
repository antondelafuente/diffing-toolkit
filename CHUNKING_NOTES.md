# Chunking Strategy Notes

## Current Limitation
The current implementation has one issue: each chunk overwrites the previous chunk's activations because they use the same output directory structure:
```
/storage/activations/{model_name}/{dataset_name}/{split}/
```

## Quick Fix Options

### Option 1: Rename directories after each chunk
After each chunk completes, manually rename the output:
```bash
# After chunk 1
mv storage/activations/gemma-3-1b-it storage/activations/gemma-3-1b-it-chunk1

# After chunk 2  
mv storage/activations/gemma-3-1b-it storage/activations/gemma-3-1b-it-chunk2
```

### Option 2: Modify the activation store directory temporarily
Before running each chunk, modify the config to use a different directory:
```bash
# In configs/config.yaml or configs/preprocessing/default.yaml
preprocessing:
  activation_store_dir: ${infrastructure.storage.base_dir}/activations_chunk1
```

### Option 3: Process different datasets per chunk
Instead of chunking the same dataset, process different datasets/splits:
- Chunk 1: tulu-3-sft train (first 1000)
- Chunk 2: fineweb train (first 1000)
- Chunk 3: tulu-3-sft validation
- etc.

## Recommended Approach for One-Time Processing

Since you only need to do this once or twice for the 7B models:

1. **For testing (small runs)**: Just let chunks overwrite - you're testing the memory usage
2. **For production (full dataset)**:
   - Run chunk 1
   - Manually copy/rename the activation directory
   - Run chunk 2
   - Manually copy/rename
   - Continue...
   
3. **Alternative**: Process different dataset types separately:
   ```bash
   # Run 1: Chat datasets only
   python main.py preprocessing.chat_only=true ...
   
   # Run 2: Pretraining datasets only  
   python main.py preprocessing.pretraining_only=true ...
   ```

## Memory Guidelines

Based on our testing:
- **1B model + 1000 samples**: ~15GB peak memory → Safe on 24GB GPU
- **7B model + 500 samples**: Expected ~20GB peak → Use 500 sample chunks for 7B
- **7B model + 1000 samples**: Might exceed 24GB → Risky

## Usage Examples

```bash
# For 1B model (safe with 1000 samples)
python run_chunked_preprocessing.py --organism caps --model gemma3_1B --chunk-size 1000

# For 7B model (use smaller chunks)
python run_chunked_preprocessing.py --organism caps --model gemma_7B --chunk-size 500

# Dry run to see what would be executed
python run_chunked_preprocessing.py --organism caps --model gemma_7B --chunk-size 500 --dry-run

# Resume from chunk 5 after a failure
python run_chunked_preprocessing.py --organism caps --model gemma_7B --chunk-size 500 --start-chunk 4
```