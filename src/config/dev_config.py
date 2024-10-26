"""
Development configuration optimized for rapid iteration and debugging.

Estimated:
- Runtime: ~5 minutes
- GPU Memory: ~4GB
- Dataset: 100 samples
- Use Case: Development iterations, debugging, feature testing

Resource Requirements:
- Entry-level GPU (2GB+ VRAM)
- 8GB system memory recommended
"""

from .base_config import BaseConfig

class DevConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Model Configuration
        self.model_name = "t5-small"  # 60M parameters - good for development
        self.max_length = 32          # Shorter sequences for faster processing

        # Dataset Size
        self.sample_size = 100        # Small but meaningful sample
        self.val_sample_size = 20     # Much smaller validation set
        self.batch_size = 8           # Larger batches for faster processing
        self.epochs = 2               # Quick iterations

        # Basic Generation
        self.beam_size = 1            # No beam search in dev
        self.do_sample = False        # Deterministic for faster generation
        self.temperature = 1.0        # No temperature sampling
        self.repetition_penalty = 1.0  # No penalty
        self.top_k = None             # Disable top_k
        self.top_p = None             # Disable top_p
        self.early_stopping = False   # Disable early stopping in generation

        # Training Features
        self.gradient_accumulation_steps = 1
        self.warmup_steps = 0         # No warmup needed for dev
        self.learning_rate = 5e-4     # Faster learning for development
        self.gradient_checkpointing = False  # Disable for faster training

        # Development Features
        self.save_checkpoints = False  # Skip checkpoint saving in dev
        self.checkpoint_frequency = 0  # No checkpoints
        self.early_stopping = False    # No early stopping in dev
        self.patience = 0             # No patience needed

        # Validation
        self.val_split = 0.2
        self.test_split = 0.0         # No test set in dev
        self.eval_frequency = 1       # Only evaluate at end of epoch
        self.eval_batches = 5         # Limit evaluation batches

        # System
        self.num_workers = 2          # Some parallel processing
        self.pin_memory = True        # Basic memory optimization

        # Logging
        self.log_every_n_steps = 50   # Reduced logging frequency
        self.sample_generations = True # Show samples during training
        self.max_samples_per_epoch = 2 # Limit sample generations

        # Performance Optimization
        self.use_fp16 = True          # Use mixed precision
        self.optimize_memory_use = True # Enable memory optimizations
        self.eval_subset_size = 20    # Small subset for validation
