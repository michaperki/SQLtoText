"""
Staging configuration for validation with moderate dataset size.

Estimated:
- Runtime: ~30 minutes
- GPU Memory: ~6GB
- Dataset: 1000 samples
- Use Case: Model validation, hyperparameter tuning, pre-production testing

Resource Requirements:
- Mid-range GPU (4GB+ VRAM)
- 16GB system memory recommended
"""

from .base_config import BaseConfig

class StagingConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Model Configuration
        self.model_name = "t5-base"   # 220M parameters - full model
        self.max_length = 128         # Standard sequence length

        # Dataset Size
        self.sample_size = 1000       # Moderate dataset
        self.batch_size = 8           # Balanced batch size
        self.epochs = 5               # Multiple epochs for validation

        # Generation Features
        self.beam_size = 2            # Basic beam search
        self.do_sample = True
        self.temperature = 0.8        # More focused sampling
        self.repetition_penalty = 1.5
        self.top_k = 50
        self.top_p = 0.95

        # Training Optimizations
        self.gradient_accumulation_steps = 2
        self.warmup_steps = 100
        self.learning_rate = 2e-4
        self.gradient_checkpointing = True  # Memory optimization

        # Validation Features
        self.save_checkpoints = True
        self.checkpoint_frequency = 1
        self.early_stopping = True
        self.patience = 3

        # Data Splits
        self.train_split = 0.7        # Proper validation splits
        self.val_split = 0.15
        self.test_split = 0.15

        # System
        self.num_workers = 2          # Parallel processing
        self.pin_memory = True        # Memory optimization enabled

        # Logging
        self.log_every_n_steps = 50
        self.sample_generations = True
