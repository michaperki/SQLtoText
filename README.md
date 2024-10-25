# SQL-to-Text Model

A deep learning project that translates SQL queries into natural language descriptions using the T5 transformer model. This model helps bridge the gap between technical SQL queries and human-readable explanations.

## Features

- Converts SQL queries to natural language using T5 transformer
- Supports beam search generation with configurable parameters
- Includes training progress visualization
- Provides model checkpoint saving and loading
- Implements early stopping and learning rate scheduling
- Supports GPU acceleration when available

## Prerequisites

- Python 3.11 or newer
- CUDA-compatible GPU (optional, but recommended for faster training)
- 8GB+ RAM recommended
- Internet connection for downloading model weights and datasets

## Installation

### 1. Set Up Python Environment

First, verify your Python installation:

```bash
python --version  # Should show Python 3.11 or newer
```

If Python isn't installed, [download it from python.org](https://www.python.org/downloads/).

> **Important**: During installation, check "Add Python to PATH"

### 2. Clone the Repository

Using GitHub Desktop:
1. Install [GitHub Desktop](https://desktop.github.com/)
2. Go to File > Clone Repository
3. Enter the repository URL
4. Choose your local path
5. Click "Clone"

Using Git command line:
```bash
git clone <repository-url>
cd sql-to-text-model
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

With your virtual environment activated, install all required packages using:

```bash
# Install all dependencies
pip install -r requirements.txt

# If you need CUDA support, also run:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: The PyTorch CUDA installation is separate since it depends on your specific hardware setup. Skip this step if you don't have a CUDA-compatible GPU.

## Configuration

The model's behavior can be customized through the `config` dictionary in `main.py`:

```python
config = {
    "model_name": "t5-small",        # Base model to use
    "batch_size": 16,                # Training batch size
    "max_length": 256,               # Maximum sequence length
    "learning_rate": 3e-5,           # Learning rate for optimization
    "epochs": 10,                    # Number of training epochs
    "beam_size": 5,                  # Beam search size for generation
    "temperature": 0.6,              # Sampling temperature
    "sample_size": 3000              # Training dataset size
    # ... (see main.py for full configuration)
}
```

## Usage

### Training the Model

1. Activate your virtual environment (if not already active)
2. Run the training script:
   ```bash
   python main.py
   ```

The script will:
- Load and preprocess the WikiSQL dataset
- Train the model for the specified number of epochs
- Save checkpoints and training metrics
- Generate a training loss plot
- Save model predictions to `predictions.txt`

### Project Structure and Outputs

```
sql-to-text-model/
├── main.py              # Main training script
├── config_minimal.py    # Minimal configuration for testing
├── cache/              # Cached tokenized datasets for faster loading
├── figures/           # Training loss plots saved as PNG files
├── logs/             # Timestamped training logs
├── model/            # Saved model checkpoints
├── predictions.txt     # Generated predictions
└── run_data.json      # Training metrics and history
```

### Output Files
- **Training Logs**: Detailed logs of each training run in `logs/training_log_[timestamp].txt`
- **Training Plots**: Loss visualization plots in `figures/training_loss_[timestamp].png`
- **Run History**: Training metrics and configurations in `run_data.json`
- **Model Output**: Generated SQL-to-text translations in `predictions.txt`

### Monitoring Progress
During training, you can monitor:
- Training loss per epoch in logs and real-time console output
- Validation loss after training completion
- Generated predictions saved to predictions.txt
- Training progress visualization saved to figures directory

## Dataset Caching

The script implements dataset caching to speed up training:
- Preprocessed datasets are saved in the `cache/` directory
- Subsequent runs with the same sample size will reuse cached data
- Cache is invalidated if sample size or other preprocessing parameters change

## Running Quick Tests

For rapid testing and development:
1. Set `USE_MINIMAL_CONFIG = True` in main.py to use minimal configuration
2. The minimal config runs with:
   - 1 epoch
   - 100 sample size
   - Larger batch size (32)
   - Reduced max sequence length
   - Minimal beam search parameters

This allows quick validation of changes while still testing the full pipeline.

## Advanced Features

### Model Checkpointing

The model automatically saves checkpoints during training. To resume from a checkpoint, set:
```python
config["reuse_model"] = True
```

### GPU Acceleration

GPU support is automatically detected and enabled when available. Monitor GPU usage with:
```bash
nvidia-smi
```

### Early Stopping

The model implements early stopping to prevent overfitting. Configure via:
```python
config["early_stopping"] = True
```

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Reduce `max_length` in config
   - Use a smaller model variant (e.g., "t5-small")

2. **Slow Training**
   - Enable GPU support
   - Increase `batch_size` if memory allows
   - Reduce `sample_size` for faster iterations

3. **Poor Generation Quality**
   - Adjust `beam_size` and `temperature`
   - Increase training epochs
   - Use a larger model variant

## Contributing

### Quick Start
1. Pull the latest changes:
   ```bash
   git pull origin main
   ```

2. Make your changes to the code

3. Push your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin main
   ```

### Development Guidelines
- Pull before making changes to avoid conflicts
- Keep commits focused on single changes when possible
- Write clear commit messages describing what you changed
- Update requirements.txt if you add new dependencies
- Update documentation if you change functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers library
- WikiSQL dataset
- PyTorch team

## Support

For issues and questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the maintainers

---
