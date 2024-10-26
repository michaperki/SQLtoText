
# src/config/prod_config.py
"""
Production configuration optimized for best model performance.
"""

from .base_config import BaseConfig

class ProdConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Full dataset
        self.sample_size = -1  # Use entire dataset
        self.batch_size = 16
        self.epochs = 10

        # Maximum sequence length
        self.max_length = 256

        # Optimal generation parameters
        self.beam_size = 3
        self.do_sample = True
        self.temperature = 0.8
        self.repetition_penalty = 2.5
        self.top_k = 50
        self.top_p = 0.92

        # Training optimizations
        self.gradient_accumulation_steps = 4
        self.warmup_steps = 500
        self.learning_rate = 1e-4

        # Thorough validation
        self.save_checkpoints = True
        self.checkpoint_frequency = 1
        self.early_stopping = True
        self.patience = 4
        self.min_delta = 0.0001

        # Proper dataset splits
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1

        # System optimizations
        self.num_workers = 4
        self.pin_memory = True
