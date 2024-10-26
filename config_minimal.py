
"""Minimal configuration for quick testing of the SQL-to-text pipeline."""

import logging
import time
import torch

minimal_config = {
    "model_name": "t5-small",              # Smaller model for faster iteration
    "batch_size": 16,                      # Smaller batch size if memory is an issue
    "max_length": 64,                      # Reduced sequence length for speed
    "learning_rate": 3e-5,
    "epochs": 1,                           # Single epoch for quick results
    "gradient_accumulation_steps": 1,      # No accumulation to reduce compute time
    "beam_size": 1,                        # Simple greedy decoding for testing
    "repetition_penalty": 1.5,
    "do_sample": True,                     # Sampling enabled for diverse outputs
    "temperature": 0.8,                    # Increased for output variety
    "early_stopping": False,
    "top_k": 30,                           # Lowered for simplified diversity
    "top_p": 0.9,                          # Top-p sampling for slight randomness
    "gpu_available": torch.cuda.is_available(),
    "sample_size": 50,                     # Minimal sample for quick testing
    "cache_dir": "./cache",
    "model_dir": "./model",
    "logging_level": logging.INFO,
    "warmup_steps": 5,                     # Minimal warmup for fast startup
    "reuse_model": False,                  # Fresh start for the test
    "run_name": f"minimal_test_{time.strftime('%Y%m%d_%H%M%S')}",
    "figures_dir": "./figures",
    "logs_dir": "./logs",
    "run_data_path": "run_data.json",
    "eval_output_file": "predictions.txt",
    "save_checkpoints": False,             # Skip checkpoints for faster runs
    "patience": 2,                         # Reduced patience to quickly stop training if no improvement
    "min_delta": 0.01                      # Adjusted min delta for quick response to changes
}

