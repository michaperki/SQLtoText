"""
Production configuration optimized for best model performance.

Estimated:
- Runtime: ~3-4 hours
- GPU Memory: ~8GB
- Dataset: Full dataset
- Use Case: Final training, production model generation

Resource Requirements:
- High-end GPU (8GB+ VRAM)
- 32GB system memory recommended
- SSD storage recommended
"""

from .base_config import BaseConfig

class ProdConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Model Configuration
        self.model_name = "t5-base"   # 220M parameters - full model
        self.max_length = 256         # Maximum sequence length

        # Dataset Size
        self.sample_size = -1         # Full dataset
        self.batch_size = 16
        self.epochs = 10

        # Generation Features
        self.beam_size = 3            # Full beam search
        self.do_sample = True
        self.temperature = 0.7        # Focused sampling
        self.repetition_penalty = 2.5
        self.top_k = 50
        self.top_p = 0.92

        # Training Optimizations
        self.gradient_accumulation_steps = 4
        self.warmup_steps = 500
        self.learning_rate = 1e-4
        self.gradient_checkpointing = True

        # Full Validation
        self.save_checkpoints = True
        self.checkpoint_frequency = 1
        self.early_stopping = True
        self.patience = 4
        self.min_delta = 0.0001      # Strict improvement threshold

        # Data Splits
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1

        # System
        self.num_workers = 4          # Maximum parallel processing
        self.pin_memory = True

        # Logging
        self.log_every_n_steps = 100
        self.sample_generations = True
