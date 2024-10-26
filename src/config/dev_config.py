# src/config/dev_config.py
"""
Development configuration optimized for fast iteration and debugging.
"""

from .base_config import BaseConfig

class DevConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Minimal dataset for rapid development
        self.sample_size = 100
        self.batch_size = 4
        self.epochs = 2

        # Shorter sequences for faster training
        self.max_length = 64

        # Basic generation parameters
        self.beam_size = 1
        self.do_sample = False
        self.temperature = 1.0

        # Simplified training
        self.gradient_accumulation_steps = 1
        self.warmup_steps = 10
        self.learning_rate = 5e-4

        # Enable debugging features
        self.save_checkpoints = True
        self.checkpoint_frequency = 1
        self.early_stopping = True
        self.patience = 2

        # Minimal validation
        self.val_split = 0.2
        self.test_split = 0.0  # No test set needed in dev

        # System optimizations
        self.num_workers = 0  # Easier debugging
        self.pin_memory = False
