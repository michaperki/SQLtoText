"""Minimal configuration for quick testing of the SQL-to-text pipeline."""

import logging
import time
import torch

minimal_config = {
    "model_name": "t5-small",
    "batch_size": 32,         # Larger batch size for fewer iterations
    "max_length": 128,        # Reduced from 256 to speed up processing
    "learning_rate": 3e-5,
    "epochs": 1,             # Just 1 epoch to test the pipeline
    "gradient_accumulation_steps": 1,  # No gradient accumulation for testing
    "beam_size": 2,          # Minimal beam search
    "repetition_penalty": 1.5,
    "temperature": 0.6,
    "early_stopping": True,
    "gpu_available": torch.cuda.is_available(),
    "sample_size": 100,      # Very small sample just to test pipeline
    "cache_dir": "./cache",
    "model_dir": "./model",
    "logging_level": logging.INFO,
    "warmup_steps": 10,      # Minimal warmup
    "reuse_model": False,
    "run_name": f"minimal_test_{time.strftime('%Y%m%d_%H%M%S')}",
    "figures_dir": "./figures",
    "logs_dir": "./logs",      # Directory for saving log files
    "run_data_path": "run_data.json",
    "eval_output_file": "predictions.txt",
    "save_checkpoints": True,
}
