
"""Intermediate configuration for a balanced SQL-to-text test."""

import logging
import time
import torch

intermediate_config = {
    "model_name": "t5-small",               # Smaller model for manageable speed
    "batch_size": 8,                        # Moderate batch size for balanced speed and stability
    "max_length": 128,                      # Allows longer sentences but not excessive
    "learning_rate": 5e-5,                  # Slightly increased to hasten convergence
    "epochs": 2,                            # Two epochs for observing potential improvement
    "gradient_accumulation_steps": 2,       # Slight accumulation for stability without heavy compute
    "beam_size": 2,                         # Small beam search for variety
    "repetition_penalty": 1.7,
    "do_sample": True,
    "temperature": 0.7,                     # Lower temperature to focus on relevant predictions
    "early_stopping": True,
    "top_k": 40,                            # Slightly higher top-k for diversity
    "top_p": 0.92,                          # Modest nucleus sampling to maintain variation
    "gpu_available": torch.cuda.is_available(),
    "sample_size": 200,                     # Larger sample size for a better assessment
    "cache_dir": "./cache",
    "model_dir": "./model",
    "logging_level": logging.INFO,
    "warmup_steps": 50,                     # Extended warmup for model stability
    "reuse_model": False,                   # Fresh start for testing consistency
    "run_name": f"intermediate_test_{time.strftime('%Y%m%d_%H%M%S')}",
    "figures_dir": "./figures",
    "logs_dir": "./logs",
    "run_data_path": "run_data.json",
    "eval_output_file": "predictions.txt",
    "save_checkpoints": True,               # Save checkpoints in case of promising performance
    "patience": 3,                          # Allows for slight patience during training
    "min_delta": 0.001                      # Stricter min delta for refined training stops
}
