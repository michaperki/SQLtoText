"""
Main entry point for the SQL to Question training pipeline.
"""

import sys
import time
import logging
import torch
from pathlib import Path
import argparse
from datetime import datetime
from typing import Tuple

# Add project root to path
src_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(src_path))

# Local imports
from config.config_factory import ConfigFactory, Environment
from utils.logging_utils import LoggerFactory
from data.data_handler import SQLDataProcessor
from model.model_handler import ModelHandler
from training.trainer import Trainer

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model to convert SQL queries to natural language questions."
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["min", "dev", "staging", "prod"],
        default="dev",
        help="Environment to run in (min/dev/staging/prod). 'min' is fastest possible execution."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to resume from",
        default=None
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()

def setup_training() -> Tuple[object, object, object, object]:
    """
    Set up all components needed for training.

    Returns:
        Tuple containing config, model handler, data processor, and trainer
    """
    # Get configuration
    args = parse_args()
    config = ConfigFactory.get_config(args.env)

    if args.debug:
        config.sample_size = 100
        config.epochs = 1
        config.logging_level = logging.DEBUG

    # Setup logging
    logger = LoggerFactory.create_logger(config)
    logger.info(f"Starting training run in {args.env} environment")

    try:
        # Initialize model handler
        model_handler = ModelHandler(config)
        tokenizer, model = model_handler.load_model_and_tokenizer()

        # Load from checkpoint if specified
        if args.checkpoint:
            model_handler.load_checkpoint(args.checkpoint)
            logger.info(f"Resumed from checkpoint: {args.checkpoint}")

        # Initialize data processor
        data_processor = SQLDataProcessor(config, tokenizer)

        # Initialize trainer
        trainer = Trainer(config, model_handler, data_processor)

        return config, model_handler, data_processor, trainer

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}", exc_info=True)
        raise

def run_training(
    config,
    model_handler,
    data_processor,
    trainer
) -> None:
    """Run the training pipeline."""
    start_time = time.time()
    logger = logging.getLogger(__name__)

    try:
        # Load and preprocess data
        logger.info("Preparing datasets...")
        train_loader, eval_loader, column_names = data_processor.load_and_process_dataset()

        # Train model
        logger.info("Starting training...")
        train_losses, final_val_loss = trainer.train(train_loader, eval_loader)

        # Save final model
        logger.info("Saving final model...")
        model_handler.save_model(suffix="final")

        # Log training summary
        total_time = time.time() - start_time
        logger.info(
            f"Training completed successfully!\n"
            f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n"
            f"Final validation loss: {final_val_loss:.4f}"
        )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save emergency backup
        model_handler.save_model(suffix="interrupted")
        raise

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        # Save emergency backup
        model_handler.save_model(suffix="error")
        raise

def main():
    """Main entry point."""
    try:
        # Setup all components
        config, model_handler, data_processor, trainer = setup_training()

        # Run training
        run_training(config, model_handler, data_processor, trainer)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        sys.exit(0)

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
