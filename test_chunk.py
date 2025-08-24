#!/usr/bin/env python3
"""
Test script to find optimal chunk size for activation extraction
without running out of memory.
"""

import sys
sys.path.append(".")

import torch
import gc
from pathlib import Path
from loguru import logger
from src.pipeline.activation_collection import collect_activations
from src.utils.configs import ModelConfig
from datasets import load_dataset

# Configure logging
logger.add(sys.stderr, level="INFO")

def test_chunk(max_samples: int, batch_size: int):
    """Test activation extraction with specific chunk size."""
    
    # Model configuration - using Gemma 3 1B (smaller model)
    model_cfg = ModelConfig(
        model_id="google/gemma-2-2b-it",  # Using 2B model which should be available
        base_model_id=None,
        adapter_id=None,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        steering_vector=None,
        steering_layer=None,
        tokenizer_id=None,
        no_auto_device_map=False,
        trust_remote_code=True
    )
    
    # Load a simple dataset - using tulu-3-sft-olmo-2-mixture
    logger.info(f"Loading dataset with max {max_samples} samples...")
    dataset = load_dataset("allenai/tulu-3-sft-olmo-2-mixture", split="train")
    
    # Limit to chunk size
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Setup output directory
    output_dir = Path("/workspace/diffing-toolkit/test_chunks")
    output_dir.mkdir(exist_ok=True)
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Monitor initial memory
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")
    
    try:
        # Run activation collection
        logger.info(f"Starting activation collection with batch_size={batch_size}")
        collect_activations(
            model_cfg=model_cfg,
            dataset=dataset,
            layers=[12],  # Just middle layer
            activation_store_dir=str(output_dir),
            dataset_name="test_chunk",
            dataset_split="train",
            max_samples=max_samples,
            max_tokens=max_samples * 1024,  # Assuming ~1024 tokens per sample
            batch_size=batch_size,
            context_len=1024,
            dtype=torch.bfloat16,
            store_tokens=False,  # Don't store tokens to save memory
            overwrite=True,
            disable_multiprocessing=True,  # Simpler for testing
            text_column=None,
            messages_column="messages",
            is_chat_data=True,
            ignore_first_n_tokens=0,
            token_level_replacement=None,
            default_text_column="text"
        )
        
        # Check final memory
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Final GPU memory: {final_memory:.2f} GB")
            logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")
            logger.info(f"Memory leaked: {(final_memory - initial_memory):.2f} GB")
        
        logger.success(f"✅ Successfully processed {max_samples} samples with batch_size={batch_size}")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ OOM with {max_samples} samples, batch_size={batch_size}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error with {max_samples} samples: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Test different chunk sizes
    test_configs = [
        # (max_samples, batch_size)
        (100, 8),      # Very small test
        (500, 8),      # Small chunk
        (1000, 8),     # Medium chunk
        (2000, 8),     # Larger chunk
        (5000, 8),     # Large chunk
    ]
    
    for max_samples, batch_size in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing chunk size: {max_samples} samples, batch_size={batch_size}")
        logger.info(f"{'='*60}")
        
        success = test_chunk(max_samples, batch_size)
        
        if not success:
            logger.warning(f"Found memory limit at {max_samples} samples")
            if max_samples > 100:
                # Try smaller size
                smaller_size = max_samples // 2
                logger.info(f"Trying smaller size: {smaller_size} samples")
                test_chunk(smaller_size, batch_size)
            break
        
        # Clear memory between tests
        torch.cuda.empty_cache()
        gc.collect()