# Model Output Directory

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
