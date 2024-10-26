"""
Base configuration file for ML training pipeline.
Contains all shared parameters and their default values.
"""

import os
import torch
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from pathlib import Path

@dataclass
class BaseConfig:
    """
    Base configuration class containing all shared parameters.
    All specific configurations (minimal, intermediate, production) will inherit from this.
    """

    def __init__(self):
        # Project Structure
        self.project_root = Path(__file__).parent.parent.parent.absolute()

        # Run Identification
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"run_{timestamp}"

        # Directories
        self.cache_dir = os.path.join(self.project_root, "cache")
        self.model_dir = os.path.join(self.project_root, "models")
        self.logs_dir = os.path.join(self.project_root, "logs")
        self.figures_dir = os.path.join(self.project_root, "figures")
        self.data_dir = os.path.join(self.project_root, "data")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")

        # File Paths
        self.run_data_path = os.path.join(self.logs_dir, "run_data.json")
        self.eval_output_file = os.path.join(self.logs_dir, f"predictions_{timestamp}.txt")

        # Model Parameters
        self.model_name = "t5-base"
        self.max_length = 128
        self.beam_size = 1
        self.batch_size = 8
        self.learning_rate = 1e-4

        # Training Parameters
        self.epochs = 3
        self.gradient_accumulation_steps = 1
        self.warmup_steps = 100
        self.early_stopping = True
        self.patience = 3
        self.min_delta = 0.001

        # Generation Parameters
        self.repetition_penalty = 1.0
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.do_sample = False

        # Dataset Parameters
        self.sample_size = 100
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1

        # System Parameters
        self.gpu_available = torch.cuda.is_available()
        self.seed = 42
        self.num_workers = 4
        self.pin_memory = True
        self.logging_level = logging.INFO

        # Model Loading/Saving
        self.reuse_model = False
        self.save_checkpoints = True
        self.checkpoint_frequency = 1

        # Create necessary directories
        self._setup_directories()

    def _setup_directories(self):
        """Creates all necessary directories for the project."""
        directories = [
            self.cache_dir,
            self.model_dir,
            self.logs_dir,
            self.figures_dir,
            self.data_dir,
            self.checkpoint_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def update(self, **kwargs):
        """
        Update configuration parameters with new values.

        Args:
            **kwargs: Keyword arguments of parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")

    def __str__(self):
        """Returns a formatted string of all configuration parameters."""
        attrs = [f"{key}={value}" for key, value in vars(self).items()
                if not key.startswith('_')]
        return f"Configuration(\n  " + "\n  ".join(attrs) + "\n)"

    def save(self, filepath: str):
        """
        Save configuration to a JSON file.

        Args:
            filepath: Path to save the configuration
        """
        import json

        config_dict = {k: v for k, v in vars(self).items()
                      if not k.startswith('_')}

        # Convert Path objects to strings
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'BaseConfig':
        """
        Load configuration from a JSON file.

        Args:
            filepath: Path to load the configuration from

        Returns:
            BaseConfig: Loaded configuration object
        """
        import json

        config = cls()

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config
