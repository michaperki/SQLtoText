
# SQL-to-Text Model

A deep learning project that translates SQL queries into natural language descriptions using the T5 transformer model. This model helps bridge the gap between technical SQL queries and human-readable explanations.

## Quick Start

### Prerequisites

- Python 3.11 or newer
- CUDA-compatible GPU (optional, but recommended)
- 8GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sql-to-text
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt

# For CUDA support (optional):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

Run training with different configurations:
```bash
# Minimal run (fastest, for testing)
python src/main.py --env min

# Development run (small dataset, fast iterations)
python src/main.py --env dev

# Staging run (medium dataset, balanced settings)
python src/main.py --env staging

# Production run (full dataset, optimal settings)
python src/main.py --env prod

# Debug mode (available for any environment)
python src/main.py --env prod --debug
```

### Project Structure
```
sql-to-text/
├── src/                # Source code
│   ├── config/        # Configuration files
│   ├── data/          # Data handling
│   ├── model/         # Model operations
│   ├── training/      # Training logic
│   ├── utils/         # Utilities
│   ├── main.py        # Entry point
│   └── README.md      # Detailed documentation
├── tests/             # Test files
├── logs/              # Training logs
├── cache/             # Dataset cache
├── models/            # Saved models
├── figures/           # Training plots
├── requirements.txt   # Dependencies
└── README.md         # This file
```

For detailed documentation about the implementation, configuration options, and advanced features, please see [src/README.md](src/README.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers library
- WikiSQL dataset
- PyTorch team

