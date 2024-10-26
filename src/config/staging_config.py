# src/config/staging_config.py
"""
Staging configuration for validation with moderate dataset size.
"""

from .base_config import BaseConfig

class StagingConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Moderate dataset size
        self.sample_size = 1000
        self.batch_size = 8
        self.epochs = 5

        # Standard sequence length
        self.max_length = 128

        # More sophisticated generation
        self.beam_size = 2
        self.do_sample = True
        self.temperature = 0.9
        self.repetition_penalty = 1.5
        self.top_k = 50
        self.top_p = 0.95

        # Training optimizations
        self.gradient_accumulation_steps = 2
        self.warmup_steps = 100
        self.learning_rate = 2e-4

        # Validation settings
        self.save_checkpoints = True
        self.checkpoint_frequency = 1
        self.early_stopping = True
        self.patience = 3

        # Proper split for validation
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15

        # System optimizations
        self.num_workers = 2
        self.pin_memory = True
