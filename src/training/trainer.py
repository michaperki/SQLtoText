"""
Training module for managing the training lifecycle and evaluation.
"""

import logging
import torch
from tqdm import tqdm
import os
from typing import Tuple, List, Dict
from torch.amp import autocast  # Updated import
from datetime import datetime

logger = logging.getLogger(__name__)

class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, config, model_handler, data_processor):
        """
        Initialize the trainer.

        Args:
            config: Configuration object
            model_handler: ModelHandler instance for model operations
            data_processor: DataProcessor instance for data operations
        """
        self.config = config
        self.model_handler = model_handler
        self.data_processor = data_processor
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []

    def train(self, train_dataloader, eval_dataloader) -> Tuple[List[float], float]:
        """
        Main training loop.

        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data

        Returns:
            Tuple[List[float], float]: List of training losses and final validation loss
        """
        logger.info("Starting training process...")

        # Setup optimization
        num_training_steps = len(train_dataloader) * self.config.epochs
        optimizer, scheduler, scaler = self.model_handler.setup_optimization(num_training_steps)

        # Outer progress bar for epochs
        epoch_bar = tqdm(
            range(self.config.epochs),
            desc="Training Progress",
            unit="epoch",
            position=0,
            colour="blue",
            leave=True
        )

        for epoch in epoch_bar:
            # Train for one epoch
            epoch_loss = self._train_epoch(
                epoch,
                train_dataloader,
                optimizer,
                scheduler,
                scaler
            )

            # Evaluate
            val_loss = self.evaluate(eval_dataloader)

            # Update progress bars
            epoch_bar.set_postfix({"train_loss": f"{epoch_loss:.4f}",
                                 "val_loss": f"{val_loss:.4f}"})

            # Early stopping check
            if not self._check_early_stopping(val_loss):
                logger.info("Early stopping triggered")
                break

            # Save checkpoint if needed
            if self.config.save_checkpoints:
                self.model_handler.save_checkpoint(epoch, val_loss)

        # Get final validation loss
        final_val_loss = self.evaluate(eval_dataloader)

        return self.train_losses, final_val_loss

    def _train_epoch(self, epoch: int, train_dataloader, optimizer,
                    scheduler, scaler) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            train_dataloader: DataLoader for training data
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            scaler: Gradient scaler

        Returns:
            float: Average loss for the epoch
        """
        logger.info(f"Starting epoch {epoch + 1}/{self.config.epochs}...")
        self.model_handler.model.train()
        epoch_loss = 0

        batch_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            unit="batch",
            position=1,
            leave=False,
            colour="green",
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                      "[⏱️ {elapsed}<{remaining}] 📉 Loss: {postfix}"
        )

        for i, batch in batch_bar:
            batch = self.model_handler.move_batch_to_device(batch)

            # Updated autocast usage
            with autocast(device_type='cuda' if self.config.gpu_available else 'cpu', dtype=torch.float16):
                outputs = self.model_handler.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                scaler.scale(loss).backward()

            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()
            batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (i + 1) % 500 == 0:
                self._generate_sample(batch)

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        self.train_losses.append(avg_epoch_loss)
        logger.info(f"Epoch {epoch + 1} completed. Avg train loss: {avg_epoch_loss:.4f}")

        return avg_epoch_loss

    def evaluate(self, eval_dataloader) -> float:
        """
        Evaluate the model.

        Args:
            eval_dataloader: DataLoader for evaluation data

        Returns:
            float: Average validation loss
        """
        logger.info("Starting model evaluation...")
        self.model_handler.model.eval()
        total_loss = 0

        eval_bar = tqdm(
            enumerate(eval_dataloader),
            total=len(eval_dataloader),
            desc="Evaluating",
            unit="batch",
            colour="yellow",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                      "[⏱️ {elapsed}<{remaining}] 📊 Loss: {postfix}",
            leave=False
        )

        predictions_file = os.path.join(
            self.config.logs_dir,
            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(predictions_file, "w", encoding="utf-8") as f:
            for i, batch in eval_bar:
                batch = self.model_handler.move_batch_to_device(batch)

                # Updated autocast usage
                with torch.no_grad(), autocast(device_type='cuda' if self.config.gpu_available else 'cpu', dtype=torch.float16):
                    outputs = self.model_handler.model(**batch)
                    loss = outputs.loss.item()
                    total_loss += loss
                    eval_bar.set_postfix({"loss": f"{loss:.4f}"})

                    # Generate and save predictions
                    generated_ids = self.model_handler.model.generate(
                        batch["input_ids"],
                        max_length=self.config.max_length,
                        num_beams=self.config.beam_size,
                        repetition_penalty=self.config.repetition_penalty,
                        temperature=self.config.temperature,
                        early_stopping=self.config.early_stopping,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample
                    )

                    preds = self.model_handler.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )

                    for pred, sql in zip(preds, batch["input_ids"]):
                        sql_str = self.model_handler.tokenizer.decode(
                            sql,
                            skip_special_tokens=True
                        )
                        f.write(f"SQL: {sql_str}\nGenerated Question: {pred}\n\n")

        avg_val_loss = total_loss / len(eval_dataloader)
        logger.info(f"Validation complete! Average loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should be stopped early.

        Args:
            val_loss: Current validation loss

        Returns:
            bool: True if training should continue, False if it should stop
        """
        if val_loss < self.best_loss - self.config.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            logger.info("New best validation loss achieved!")
            return True

        self.patience_counter += 1
        if self.patience_counter >= self.config.patience:
            return False

        return True

    def _generate_sample(self, batch: Dict) -> None:
        """
        Generate and log a sample prediction during training.

        Args:
            batch: Current batch of data
        """
        self.model_handler.model.eval()
        sql_query = self.model_handler.tokenizer.decode(
            batch['input_ids'][0],
            skip_special_tokens=True
        )
        sample_input = batch['input_ids'][0].unsqueeze(0)

        with torch.no_grad():
            sample_output = self.model_handler.model.generate(
                sample_input,
                max_length=self.config.max_length,
                num_beams=self.config.beam_size
            )

        sample_text = self.model_handler.tokenizer.decode(
            sample_output[0],
            skip_special_tokens=True
        )

        logger.info(f"SQL: {sql_query}")
        logger.info(f"Generated Question: {sample_text}")
        self.model_handler.model.train()
