"""
Minimal configuration optimized for fastest possible execution and sanity checks.

Estimated:
- Runtime: ~1 minute
- GPU Memory: ~2GB
- Dataset: 10 samples
- Use Case: Quick sanity checks, testing code changes

Resource Requirements:
- Any GPU (even low-end) or CPU
- Minimal RAM (4GB system memory)
"""

from .base_config import BaseConfig

class MinConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Model Configuration
        self.model_name = "t5-small"  # 60M parameters - smallest variant
        self.max_length = 32          # Minimum viable sequence length

        # Dataset Size
        self.sample_size = 10         # Absolute minimum for testing
        self.batch_size = 2           # Tiny batches
        self.epochs = 1               # Single pass

        # Basic Generation
        self.beam_size = 1            # No beam search
        self.do_sample = False        # Deterministic generation
        self.temperature = 1.0
        self.repetition_penalty = 1.0
        self.top_k = None             # Disable top_k sampling
        self.top_p = None             # Disable top_p sampling

        # Minimal Training
        self.gradient_accumulation_steps = 1
        self.warmup_steps = 0         # No warmup needed for tiny dataset
        self.learning_rate = 1e-4
        self.gradient_checkpointing = False

        # Features (most disabled)
        self.save_checkpoints = False
        self.early_stopping = False
        self.save_model = False

        # Validation
        self.val_split = 0.2
        self.test_split = 0.0
        self.eval_batches = 1         # Minimal validation

        # System
        self.num_workers = 0          # Single thread loading
        self.pin_memory = False       # Memory optimization off

        # Logging
        self.log_every_n_steps = 999999  # Minimal logging
        self.sample_generations = False   # Skip sample generation
