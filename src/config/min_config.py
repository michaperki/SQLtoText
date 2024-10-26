"""
Minimal configuration optimized for fastest possible execution and basic testing.
"""

from .base_config import BaseConfig

class MinConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Absolute minimum dataset
        self.sample_size = 10  # Just 10 samples
        self.batch_size = 2    # Tiny batches
        self.epochs = 1        # Single epoch

        # Minimum sequence length
        self.max_length = 32   # Short sequences

        # Simplest generation parameters
        self.beam_size = 1
        self.do_sample = False
        self.temperature = 1.0
        self.repetition_penalty = 1.0
        self.top_k = 1
        self.top_p = 1.0

        # Minimal training setup
        self.gradient_accumulation_steps = 1
        self.warmup_steps = 0
        self.learning_rate = 1e-4

        # Disable extra features
        self.save_checkpoints = False
        self.early_stopping = False

        # Minimal validation
        self.val_split = 0.2
        self.test_split = 0.0

        # System optimizations
        self.num_workers = 0   # No parallel loading
        self.pin_memory = False
