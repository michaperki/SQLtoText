"""
Handles organized storage and management of model outputs and results.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import torch
import platform
import psutil

logger = logging.getLogger(__name__)

class ResultManager:
    """Manages storage and organization of model outputs and results."""

    def __init__(self, config):
        """
        Initialize ResultManager with configuration.

        Args:
            config: Configuration object containing run parameters
        """
        self.config = config
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self._setup_output_directory()
        self._save_run_metadata()
        self._save_config()

    def _setup_output_directory(self) -> Path:
        """Create and setup the output directory structure."""
        # Create main output directory under project root
        output_base = Path(self.config.project_root) / "outputs"
        run_dir = output_base / f"run_{self.run_timestamp}"

        # Create subdirectories
        subdirs = [
            "predictions",
            "plots",
            "logs",
            "checkpoints"
        ]

        for subdir in subdirs:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create README if it doesn't exist
        readme_path = output_base / "README.md"
        if not readme_path.exists():
            self._create_output_readme(readme_path)

        return run_dir

    def _create_output_readme(self, path: Path):
        """Create README file explaining output directory structure."""
        readme_content = """# Model Output Directory

This directory contains organized outputs from model training runs.

## Directory Structure

Each run creates a timestamped directory (e.g., `run_20241026_115208/`) containing:

- `config.json`: Configuration used for the run
- `metadata.json`: Run metadata (hardware info, timing, etc.)
- `metrics.json`: Training and validation metrics
- `predictions/`: Model predictions and samples
- `plots/`: Visualizations of training progress
- `logs/`: Detailed training logs
- `checkpoints/`: Model checkpoints (if enabled)

## File Formats

- JSON files contain structured data and can be loaded programmatically
- PNG files contain visualizations
- TXT files contain human-readable logs
- PT files are PyTorch model checkpoints
"""
        path.write_text(readme_content)

    def _save_run_metadata(self):
        """Save metadata about the run, including hardware info."""
        gpu_info = None
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(0)
            }

        metadata = {
            "timestamp": self.run_timestamp,
            "platform": {
                "system": platform.system(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "gpu": gpu_info
            },
            "environment": self.config.environment if hasattr(self.config, 'environment') else 'unknown'
        }

        self._save_json(metadata, "metadata.json")

    def _save_config(self):
        """Save configuration used for this run."""
        config_dict = {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(self.config).items()
                      if not k.startswith('_')}
        self._save_json(config_dict, "config.json")

    def save_predictions(self, predictions: List[Dict], split: str = "eval"):
        """
        Save model predictions.

        Args:
            predictions: List of prediction dictionaries
            split: Data split (train/eval/test)
        """
        filename = f"{split}_predictions.json"
        self._save_json(predictions, f"predictions/{filename}")

    def save_metrics(self, metrics: Dict[str, Any]):
        """
        Save training/evaluation metrics.

        Args:
            metrics: Dictionary of metrics to save
        """
        self._save_json(metrics, "metrics.json")

    def save_plot(self, figure: plt.Figure, name: str):
        """
        Save a matplotlib figure.

        Args:
            figure: matplotlib figure to save
            name: Name for the plot file
        """
        if not name.endswith('.png'):
            name += '.png'
        figure.savefig(self.output_dir / "plots" / name)
        plt.close(figure)

    def add_log_entry(self, entry: str, log_type: str = "training"):
        """
        Add entry to specified log file.

        Args:
            entry: Log entry text
            log_type: Type of log (training/error)
        """
        log_path = self.output_dir / "logs" / f"{log_type}_log.txt"
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {entry}\n")

    def save_checkpoint(self, state_dict: Dict, checkpoint_name: str):
        """
        Save a model checkpoint.

        Args:
            state_dict: Model state dictionary
            checkpoint_name: Name for the checkpoint
        """
        if not checkpoint_name.endswith('.pt'):
            checkpoint_name += '.pt'
        checkpoint_path = self.output_dir / "checkpoints" / checkpoint_name
        torch.save(state_dict, checkpoint_path)

    def _save_json(self, data: Dict, filename: str):
        """Helper method to save JSON data."""
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_run_dir(self) -> Path:
        """Get the path to the current run directory."""
        return self.output_dir
