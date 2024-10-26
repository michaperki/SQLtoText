"""
Training module with integrated result management.
"""

import logging
import torch
from tqdm import tqdm
import os
import time
from typing import Tuple, List, Dict
from torch.amp import autocast
from datetime import datetime
import matplotlib.pyplot as plt
from utils.result_manager import ResultManager

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, model_handler, data_processor):
        """Initialize trainer with result management."""
        self.config = config
        self.model_handler = model_handler
        self.data_processor = data_processor
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.start_time = None
        self.lr_history = []

        # Initialize result manager
        self.result_manager = ResultManager(config)

    def train(self, train_dataloader, eval_dataloader) -> Tuple[List[float], float]:
        """Main training loop with comprehensive result tracking."""
        logger.info("Starting training process...")
        self.result_manager.add_log_entry("Training started")
        self.start_time = time.time()

        # Setup optimization
        num_training_steps = len(train_dataloader) * self.config.epochs
        optimizer, scheduler, scaler = self.model_handler.setup_optimization(num_training_steps)

        # Training loop
        epoch_bar = tqdm(
            range(self.config.epochs),
            desc="Training Progress",
            unit="epoch",
            position=0,
            colour="blue",
            leave=True
        )

        try:
            for epoch in epoch_bar:
                self.current_epoch = epoch

                # Train one epoch
                epoch_loss = self._train_epoch(
                    epoch,
                    train_dataloader,
                    optimizer,
                    scheduler,
                    scaler
                )

                # Evaluate
                val_loss, predictions = self.evaluate(eval_dataloader)
                self.val_losses.append(val_loss)

                # Save predictions
                self.result_manager.save_predictions(predictions, f"epoch_{epoch}")

                # Track learning rate
                current_lr = optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)

                # Save metrics
                metrics = {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_time": time.time() - self.start_time,
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None
                }
                self.result_manager.save_metrics(metrics)

                # Update progress bars
                epoch_bar.set_postfix({
                    "train_loss": f"{epoch_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}"
                })

                # Early stopping check
                if not self._check_early_stopping(val_loss):
                    logger.info("Early stopping triggered")
                    break

                # Save checkpoint if needed
                if self.config.save_checkpoints:
                    self.model_handler.save_checkpoint(
                        epoch,
                        val_loss,
                        checkpoint_dir=self.result_manager.get_run_dir() / "checkpoints"
                    )

            # Final evaluation
            final_val_loss, final_predictions = self.evaluate(eval_dataloader)

            # Save final predictions
            self.result_manager.save_predictions(final_predictions, "final_predictions")

            # Save all plots
            self._save_all_plots()

            # Save final summary metrics
            self._save_final_summary(final_val_loss, optimizer)

            return self.train_losses, final_val_loss

        except Exception as e:
            error_msg = f"Training failed with error: {str(e)}"
            logger.error(error_msg)
            self.result_manager.add_log_entry(error_msg, "error")
            raise

    def _save_all_plots(self):
        """Save all training visualization plots."""
        self._save_loss_plot()
        self._save_validation_plot()
        self._save_lr_plot()

    def _save_loss_plot(self):
        """Create and save the training loss plot."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        self.result_manager.save_plot(plt.gcf(), "training_loss")
        plt.close()

    def _save_validation_plot(self):
        """Create and save the validation loss plot."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        self.result_manager.save_plot(plt.gcf(), "loss_comparison")
        plt.close()

    def _save_lr_plot(self):
        """Create and save the learning rate schedule plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.lr_history, 'g-')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        self.result_manager.save_plot(plt.gcf(), "learning_rate_schedule")
        plt.close()

    def _save_final_summary(self, final_val_loss: float, optimizer):
        """Save final training summary metrics."""
        final_metrics = {
            "total_epochs": self.current_epoch + 1,
            "best_val_loss": self.best_loss,
            "final_val_loss": final_val_loss,
            "early_stopping_triggered": self.patience_counter >= self.config.patience,
            "total_training_time": time.time() - self.start_time,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
            "final_learning_rate": optimizer.param_groups[0]['lr'],
            "timestamp": datetime.now().isoformat(),
            "training_completed": True
        }
        self.result_manager.save_metrics(final_metrics)

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
                self._generate_sample(batch, i, optimizer.param_groups[0]['lr'])

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        self.train_losses.append(avg_epoch_loss)
        logger.info(f"Epoch {epoch + 1} completed. Avg train loss: {avg_epoch_loss:.4f}")

        return avg_epoch_loss

    def evaluate(self, eval_dataloader) -> Tuple[float, List[Dict]]:
        """
        Evaluate the model.

        Args:
            eval_dataloader: DataLoader for evaluation data

        Returns:
            Tuple[float, List[Dict]]: Average validation loss and predictions
        """
        logger.info("Starting model evaluation...")
        self.model_handler.model.eval()
        total_loss = 0
        all_predictions = []

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

        for i, batch in eval_bar:
            batch = self.model_handler.move_batch_to_device(batch)

            with torch.no_grad(), autocast(device_type='cuda' if self.config.gpu_available else 'cpu', dtype=torch.float16):
                outputs = self.model_handler.model(**batch)
                loss = outputs.loss.item()
                total_loss += loss
                eval_bar.set_postfix({"loss": f"{loss:.4f}"})

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
                    prediction_item = {
                        "sql_query": sql_str,
                        "generated_question": pred,
                        "loss": loss,
                        "timestamp": datetime.now().isoformat(),
                        "epoch": self.current_epoch,
                        "batch_idx": i
                    }
                    all_predictions.append(prediction_item)

        avg_val_loss = total_loss / len(eval_dataloader)
        logger.info(f"Validation complete! Average loss: {avg_val_loss:.4f}")
        return avg_val_loss, all_predictions

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

    def _generate_sample(self, batch: Dict, batch_idx: int, current_lr: float) -> None:
        """
        Generate and log a sample prediction during training.

        Args:
            batch: Current batch of data
            batch_idx: Current batch index
            current_lr: Current learning rate
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

        sample_prediction = {
            "sql_query": sql_query,
            "generated_question": sample_text,
            "type": "training_sample",
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "batch_idx": batch_idx,
            "learning_rate": current_lr
        }
        self.result_manager.save_predictions([sample_prediction], "training_samples")

        logger.info(f"SQL: {sql_query}")
        logger.info(f"Generated Question: {sample_text}")

        self.model_handler.model.train()
