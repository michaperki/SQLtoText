import re
import torch
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import json
import matplotlib.pyplot as plt
from config_minimal import minimal_config
from config_intermediate import intermediate_config
from torch.cuda.amp import GradScaler, autocast
import colorama
from colorama import Fore, Back, Style
from datetime import datetime
import sys
import codecs

# Initialize colorama
colorama.init(autoreset=True)

logger = None


# Update ColoredFormatter to handle encoding errors gracefully
class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors and emojis to logging outputs with Windows compatibility"""

    level_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE
    }

    message_indicators = {
        "training": "🚀",
        "epoch": "📈",
        "evaluation": "🎯",
        "model": "🤖",
        "dataset": "📊",
        "time": "⏱️",
        "loss": "📉",
        "save": "💾",
        "gpu": "🎮",
        "sql": "📝",
        "question": "❓",
        "loading": "📥",
        "config": "⚙️",
        "error": "❌",
        "success": "✅",
        "memory": "🧠",
        "checkpoint": "📍",
        "progress": "🔄"
    }

    def format(self, record):
        try:
            color = self.level_colors.get(record.levelno, Fore.WHITE)
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

            # Find the most relevant emoji for the message
            emoji = ""
            matched_indicator = None
            for indicator, symbol in self.message_indicators.items():
                if indicator.lower() in record.getMessage().lower():
                    if matched_indicator is None or len(indicator) > len(matched_indicator):
                        emoji = symbol + " "
                        matched_indicator = indicator

            message = record.getMessage()

            # Enhanced color formatting for SQL and Questions
            if "SQL:" in message:
                # Only highlight the actual SQL query, not the prompt
                if "Rephrase this SQL query into a natural language question:" in message:
                    prefix, rest = message.split("Rephrase this SQL query into a natural language question:", 1)
                    message = f"{prefix}Rephrase this SQL query into a natural language question:{Back.BLUE}{Fore.WHITE}{rest}{Style.RESET_ALL}"
                else:
                    # For direct SQL logs
                    sql_parts = message.split("SQL:", 1)
                    message = f"{sql_parts[0]}SQL:{Back.BLUE}{Fore.WHITE}{sql_parts[1]}{Style.RESET_ALL}"

            elif "Generated Question:" in message:
                q_parts = message.split("Generated Question:", 1)
                message = f"{q_parts[0]}Generated Question:{Back.GREEN}{Fore.WHITE}{q_parts[1]}{Style.RESET_ALL}"

            elif "Test prompt:" in message:
                # Only highlight the actual SQL query in the test prompt
                if "Rephrase this SQL query into a natural language question:" in message:
                    prefix, rest = message.split("Rephrase this SQL query into a natural language question:", 1)
                    message = f"{prefix}Rephrase this SQL query into a natural language question:{Back.CYAN}{Fore.WHITE}{rest}{Style.RESET_ALL}"
                else:
                    p_parts = message.split("Test prompt:", 1)
                    message = f"{p_parts[0]}Test prompt:{Back.CYAN}{Fore.WHITE}{p_parts[1]}{Style.RESET_ALL}"

            return f"{color}{timestamp}{Style.RESET_ALL} - {emoji}{message}"
        except Exception:
            # Fallback formatting if encoding issues occur
            return f"{timestamp} - {record.getMessage()}"

# Production configuration
production_config = {
    "model_name": "t5-base",
    "batch_size": 2,
    "max_length": 50,
    "learning_rate": 1e-4,
    "epochs": 10,
    "gradient_accumulation_steps": 4,
    "beam_size": 3,
    "repetition_penalty": 2.5,
    "do_sample": True,
    "temperature": 1.2,
    "early_stopping": False,
    "top_k": 50,
    "top_p": 0.95,
    "gpu_available": torch.cuda.is_available(),
    "sample_size": 1000,
    "cache_dir": "./cache",
    "model_dir": "./model",
    "logging_level": logging.INFO,
    "warmup_steps": 500,
    "reuse_model": False,
    "run_name": f"run_{time.strftime('%Y%m%d_%H%M%S')}",
    "figures_dir": "./figures",
    "logs_dir": "./logs",
    "run_data_path": "run_data.json",
    "eval_output_file": "predictions.txt",
    "save_checkpoints": False,
    "patience": 4,
    "min_delta": 0.0001
}

# Choose configuration
CONFIG_CHOICE = "production"  # Options: "minimal", "intermediate", "production"

if CONFIG_CHOICE == "minimal":
    config = minimal_config
elif CONFIG_CHOICE == "intermediate":
    config = intermediate_config
else:
    config = production_config

def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def save_run_data(config, train_losses, final_val_loss):
    run_data = {
        "config": config,
        "train_losses": train_losses,
        "final_val_loss": final_val_loss,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    if os.path.exists(config["run_data_path"]):
        with open(config["run_data_path"], "r") as file:
            data = json.load(file)
    else:
        data = []

    data.append(run_data)

    with open(config["run_data_path"], "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"Run data saved successfully")

def plot_run_data(current_run, train_losses, final_val_loss):
    plt.figure(figsize=(10, 6))
    current_epochs = range(1, len(train_losses) + 1)
    plt.plot(current_epochs, train_losses, 'b-', label=f"{current_run} - Train Loss")
    plt.plot(len(train_losses), final_val_loss, 'ro', label=f"{current_run} - Final Val Loss")

    if os.path.exists(config["run_data_path"]):
        with open(config["run_data_path"], "r") as file:
            data = json.load(file)
            for run in data:
                if len(run["train_losses"]) == len(train_losses):
                    past_train_losses = run["train_losses"]
                    past_val_loss = run["final_val_loss"]
                    run_name = run["config"]["run_name"]
                    plt.plot(current_epochs, past_train_losses, linestyle='--', alpha=0.5,
                            label=f"{run_name} - Train Loss")
                    plt.plot(len(past_train_losses), past_val_loss, 'o', alpha=0.5,
                            label=f"{run_name} - Final Val Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses Across Runs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    os.makedirs(config["figures_dir"], exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(config["figures_dir"], f"training_loss_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training and validation loss plot saved to {filename}")

def enable_optimizations(model):
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing for memory optimization")

@track_time
def load_or_train_model_and_tokenizer():
    logger.info("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"], legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        config["model_dir"] if config["reuse_model"] else config["model_name"],
        use_cache=False
    )
    model.to(device)
    enable_optimizations(model)
    logger.info("Model and tokenizer loaded successfully")
    return tokenizer, model

def save_model(model):
    os.makedirs(config["model_dir"], exist_ok=True)
    model.save_pretrained(config["model_dir"])
    logger.info("Model saved to directory")

@track_time
def load_and_preprocess_dataset(tokenizer):
    os.makedirs(config["cache_dir"], exist_ok=True)
    logger.info("Loading dataset...")
    dataset = load_dataset("wikisql")
    logger.info(f"Dataset loaded with {len(dataset['train'])} training samples and {len(dataset['validation'])} validation samples")

    cache_file = os.path.join(config["cache_dir"], f"tokenized_dataset_{config['sample_size']}.arrow")

    if os.path.exists(cache_file):
        logger.info("Loading tokenized dataset from cache...")
        tokenized_dataset = DatasetDict.load_from_disk(cache_file)
    else:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(config["sample_size"]))
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(100))
        column_names = dataset['train'][0]['table']['header']

        logger.info(f"Sample column names: {column_names[:5]}... (truncated)")

        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, column_names),
            batched=True,
            remove_columns=["question"]
        )
        tokenized_dataset.save_to_disk(cache_file)
        logger.info("Tokenized dataset cached successfully")

    return tokenized_dataset, dataset['train'][0]['table']['header']

def sql_dict_to_string(sql_dict, column_names):
    try:
        select_clause = f'SELECT {column_names[sql_dict["select"][1]]}' if sql_dict["select"][1] < len(column_names) else "SELECT *"
    except KeyError:
        select_clause = "SELECT *"
    where_clause = ""
    if "conds" in sql_dict and len(sql_dict['conds']) > 0:
        conditions = [
            f'{column_names[col]} {"=" if op == 0 else "!="} "{cond}"'
            for col, op, cond in zip(
                sql_dict['conds']['column_index'],
                sql_dict['conds']['operator_index'],
                sql_dict['conds']['condition']
            )
            if col < len(column_names)
        ]
        where_clause = " AND ".join(conditions)
    query = select_clause
    if where_clause:
        query += f' WHERE {where_clause}'
    return query

def preprocess_function(examples, tokenizer, column_names):
    inputs = [f"Rephrase this SQL query into a natural language question: {sql_dict_to_string(item, column_names)}"
              for item in examples["sql"]]
    targets = examples["question"]

    model_inputs = tokenizer(
        inputs,
        max_length=config["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = tokenizer(
        targets,
        max_length=config["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    logger.info(f"SQL: {inputs[0]}")
    logger.info(f"Generated Question: {targets[0]}")

    return model_inputs

def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'labels': labels}

@track_time
def train_model(model, optimizer, scheduler, train_dataloader, tokenizer):
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler("cuda")

    # Outer progress bar for epochs
    epoch_bar = tqdm(
        range(config["epochs"]),
        desc="Training Progress",
        unit="epoch",
        position=0,
        colour="blue",
        leave=True
    )

    for epoch in epoch_bar:
        logger.info(f"Starting epoch {epoch + 1}/{config['epochs']}...")
        model.train()
        epoch_loss = 0

        # Inner progress bar for batches
        batch_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{config['epochs']}",
            unit="batch",
            position=1,
            leave=False,
            colour="green",
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [⏱️ {elapsed}<{remaining}] 📉 Loss: {postfix}"
        )

        for i, batch in batch_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda'):
                outputs = model(**batch)
                loss = outputs.loss / config["gradient_accumulation_steps"]
                scaler.scale(loss).backward()

            if (i + 1) % config["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()
            # Update batch progress bar with current loss
            batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (i + 1) % 500 == 0:
                model.eval()
                sql_query = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
                sample_input = batch['input_ids'][0].unsqueeze(0)
                with torch.no_grad():
                    sample_output = model.generate(
                        sample_input,
                        max_length=config["max_length"],
                        num_beams=config["beam_size"]
                    )
                sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                logger.info(f"SQL: {sql_query}")
                logger.info(f"Generated Question: {sample_text}")
                model.train()

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        # Update epoch progress bar with average loss
        epoch_bar.set_postfix({"avg_loss": f"{avg_train_loss:.4f}"})
        logger.info(f"Epoch {epoch + 1} completed. Avg train loss: {avg_train_loss:.4f}")

        val_loss = evaluate_model(model, eval_dataloader, column_names, tokenized_dataset, tokenizer)
        if val_loss < best_loss - config["min_delta"]:
            best_loss = val_loss
            patience_counter = 0
            logger.info("New best validation loss achieved!")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info("Early stopping triggered")
                break
        save_model(model)
    return train_losses

@track_time
def evaluate_model(model, eval_dataloader, column_names, dataset, tokenizer):
    logger.info("Starting model evaluation...")
    model.eval()
    total_loss = 0

    # Enhanced evaluation progress bar
    eval_bar = tqdm(
        enumerate(eval_dataloader),
        total=len(eval_dataloader),
        desc="Evaluating",
        unit="batch",
        colour="yellow",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [⏱️ {elapsed}<{remaining}] 📊 Loss: {postfix}",
        leave=False
    )

    with open(config["eval_output_file"], "w", encoding="utf-8") as f:
        for i, batch in eval_bar:
            if i % 50 == 0:
                logger.info(f"Evaluation progress: {i}/{len(eval_dataloader)}")

            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda"), torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss.item()
                total_loss += loss
                # Update progress bar with current loss
                eval_bar.set_postfix({"loss": f"{loss:.4f}"})

                generated_ids = model.generate(
                    batch["input_ids"],
                    max_length=config["max_length"],
                    num_beams=config["beam_size"],
                    repetition_penalty=config["repetition_penalty"],
                    temperature=config["temperature"],
                    early_stopping=config["early_stopping"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    do_sample=config["do_sample"]
                )
                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred, sql in zip(preds, batch["input_ids"]):
                sql_str = tokenizer.decode(sql, skip_special_tokens=True)
                f.write(f"SQL: {sql_str}\nGenerated Question: {pred}\n\n")

    avg_val_loss = total_loss / len(eval_dataloader)
    logger.info(f"Validation complete! Average loss: {avg_val_loss:.4f}")
    return avg_val_loss

def setup_logging(config):
    """Enhanced logging setup with colored output and emojis that works on Windows"""
    os.makedirs(config["logs_dir"], exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config["logs_dir"], f"training_log_{timestamp}.txt")

    # Force UTF-8 encoding for file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')

    # Special handling for Windows console
    if sys.platform.startswith('win'):
        # Enable Unicode output on Windows
        sys.stdout.reconfigure(encoding='utf-8')
        # Use sys.stdout directly to ensure proper encoding
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        console_handler = logging.StreamHandler()

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = ColoredFormatter()

    # Set formatters for handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(config["logging_level"])

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Initial logs
    logger.info(f"Initializing training run: {config['run_name']}")
    logger.info(f"Log file created at: {log_file}")

    return logger

if __name__ == "__main__":
    start_time = time.time()
    logger = setup_logging(config)

    # Initial setup logs
    logger.info(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    device = torch.device("cuda" if config["gpu_available"] else "cpu")

    try:
        # Load model and tokenizer
        tokenizer, model = load_or_train_model_and_tokenizer()

        # Quick inference test
        test_sql_query = "SELECT Points FROM table WHERE Tries = 55"
        prompt = f"Rephrase this SQL query into a natural language question: {test_sql_query}"
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        model.eval()
        with torch.no_grad():
            sample_output = model.generate(inputs, max_length=config["max_length"], num_beams=config["beam_size"])
        generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
        logger.info(f"Test prompt: {prompt}")
        logger.info(f"Generated text: {generated_text}")

        # Dataset loading and preprocessing
        tokenized_dataset, column_names = load_and_preprocess_dataset(tokenizer)
        train_dataloader = DataLoader(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=config["batch_size"],
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )
        eval_dataloader = DataLoader(
            tokenized_dataset["validation"],
            batch_size=config["batch_size"],
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )

        # Optimizer and scheduler setup
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=len(train_dataloader) * config["epochs"]
        )

        # Training
        logger.info("Starting training process...")
        train_losses = train_model(model, optimizer, scheduler, train_dataloader, tokenizer)

        # Final evaluation
        logger.info("Running final evaluation...")
        final_val_loss = evaluate_model(model, eval_dataloader, column_names, tokenized_dataset, tokenizer)

        # Save and plot results
        save_run_data(config, train_losses, final_val_loss)
        plot_run_data(config["run_name"], train_losses, final_val_loss)

        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
