
import re
import torch
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup, EncoderDecoderCache
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

logger = None

# Production configuration
production_config = {
    "model_name": "t5-small",
    "batch_size": 16,
    "max_length": 256,
    "learning_rate": 3e-5,
    "epochs": 10,
    "gradient_accumulation_steps": 2,
    "beam_size": 5,
    "repetition_penalty": 1.5,
    "temperature": 0.6,
    "early_stopping": True,
    "gpu_available": torch.cuda.is_available(),
    "sample_size": 3000,
    "cache_dir": "./cache",
    "model_dir": "./model",
    "logging_level": logging.INFO,
    "warmup_steps": 500,
    "reuse_model": False,
    "run_name": f"run_{time.strftime('%Y%m%d_%H%M%S')}",
    "figures_dir": "./figures",
    "logs_dir": "./logs",      # Directory for saving log files
    "run_data_path": "run_data.json",
    "eval_output_file": "predictions.txt",
    "save_checkpoints": True,
}

# Easy switch between configurations using environment variable or command line argument
USE_MINIMAL_CONFIG = False  # You can also use os.getenv('USE_MINIMAL_CONFIG', 'False').lower() == 'true'
config = minimal_config if USE_MINIMAL_CONFIG else production_config

# Function to track execution time of functions
def track_time(func):
    """Decorator for tracking execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Save run data to JSON file
def save_run_data(config, train_losses, final_val_loss):
    """Saves the configuration and loss values to a JSON file for historical comparison."""
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

# Plot run data from JSON file
def plot_run_data(current_run, train_losses, final_val_loss):
    """Plots training losses and validation loss for the current run and compares with previous runs."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    # Plot current run's training loss
    plt.plot(epochs, train_losses, 'b-', label=f"{current_run} - Train Loss")

    # Plot final validation loss point
    plt.plot(len(train_losses), final_val_loss, 'ro', label=f"{current_run} - Final Val Loss")

    # Load and plot data from previous runs
    if os.path.exists(config["run_data_path"]):
        with open(config["run_data_path"], "r") as file:
            data = json.load(file)
            for run in data:
                past_train_losses = run["train_losses"]
                past_val_loss = run["final_val_loss"]
                run_name = run["config"]["run_name"]

                # Plot past training losses with different line styles
                plt.plot(epochs, past_train_losses, linestyle='--', alpha=0.5,
                        label=f"{run_name} - Train Loss")

                # Plot past validation loss points
                plt.plot(len(past_train_losses), past_val_loss, 'o', alpha=0.5,
                        label=f"{run_name} - Final Val Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses Across Runs")

    # Adjust legend to be outside the plot to avoid overcrowding
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Create figures directory if it doesn't exist
    os.makedirs(config["figures_dir"], exist_ok=True)

    # Save the plot with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(config["figures_dir"], f"training_loss_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Training and validation loss plot saved to {filename}")

@track_time
def load_or_train_model_and_tokenizer():
    """Loads or initializes model and tokenizer. If a saved model exists, it will load it based on config."""
    logger.info("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"], legacy=False)

    # Check if a saved model exists and load it if reuse_model is set to True
    if config["reuse_model"] and os.path.exists(config["model_dir"]):
        logger.info("Loading model from saved checkpoint...")
        model = T5ForConditionalGeneration.from_pretrained(config["model_dir"])
    else:
        logger.info("Training model from scratch...")
        model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

    model.to(device)
    return tokenizer, model

# Save model after training
def save_model(model):
    """Saves the trained model to the specified directory."""
    os.makedirs(config["model_dir"], exist_ok=True)
    model.save_pretrained(config["model_dir"])
    logger.info("Model saved to directory.")

@track_time
def load_and_preprocess_dataset(tokenizer):
    """Loads and preprocesses the dataset for training, applying tokenization."""
    # Create cache directory if it doesn't exist
    os.makedirs(config["cache_dir"], exist_ok=True)

    logger.info("Loading dataset...")
    dataset = load_dataset("wikisql")

    cache_file = os.path.join(config["cache_dir"], f"tokenized_dataset_{config['sample_size']}.arrow")
    logger.info(f"Looking for cached dataset at: {cache_file}")

    if os.path.exists(cache_file):
        logger.info("Loading tokenized dataset from cache...")
        tokenized_dataset = DatasetDict.load_from_disk(cache_file)
        logger.info("Successfully loaded from cache!")
    else:
        logger.info("No cache found. Processing dataset...")
        # Limit dataset size for faster processing
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(config["sample_size"]))
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(1000))

        column_names = dataset['train'][0]['table']['header']
        logger.info(f"Available column names: {column_names}")

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, column_names),
            batched=True,
            remove_columns=["question"]
        )

        logger.info(f"Saving processed dataset to cache: {cache_file}")
        tokenized_dataset.save_to_disk(cache_file)
        logger.info("Dataset cached successfully!")

    return tokenized_dataset, dataset['train'][0]['table']['header']

def sql_dict_to_string(sql_dict, column_names):
    """Converts a SQL dictionary structure to a SQL string with column filtering."""
    try:
        select_clause = f'SELECT {column_names[sql_dict["select"][1]]}' if sql_dict["select"][1] < len(column_names) else "SELECT *"
    except KeyError:
        select_clause = "SELECT *"

    where_clause = ""
    if "conds" in sql_dict and len(sql_dict['conds']) > 0:
        conditions = []
        for col, op, cond in zip(sql_dict['conds']['column_index'], sql_dict['conds']['operator_index'], sql_dict['conds']['condition']):
            if col < len(column_names):
                column_name = column_names[col]
                operator = "=" if op == 0 else "!="
                if cond and re.match(r'^[a-zA-Z0-9\s\-]+$', cond) and len(cond) < 30:
                    conditions.append(f'{column_name} {operator} "{cond}"')

        where_clause = " AND ".join(conditions)

    query = select_clause
    if where_clause:
        query += f' WHERE {where_clause}'
    return query

def preprocess_function(examples, tokenizer, column_names):
    """Prepares model inputs and labels by tokenizing SQL and corresponding questions."""
    inputs = ["translate SQL to English: " + sql_dict_to_string(item, column_names) for item in examples["sql"]]
    targets = [item for item in examples["question"]]

    model_inputs = tokenizer(inputs, max_length=config["max_length"], truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, max_length=config["max_length"], truncation=True, padding="max_length", return_tensors="pt").input_ids

    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def collate_fn(batch):
    """Custom collate function to handle dynamic padding in dataloader."""
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {'input_ids': input_ids, 'labels': labels}

@track_time
def train_model(model, optimizer, scheduler, train_dataloader):
    """Training loop that tracks and returns training loss per epoch."""
    train_losses = []

    # Save checkpoint after each epoch
    for epoch in range(config["epochs"]):
        logger.info(f"Starting epoch {epoch + 1}/{config['epochs']}...")
        model.train()
        epoch_loss = 0

        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (i + 1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch + 1} completed. Avg train loss: {avg_train_loss:.4f}")

        # Save checkpoint after each epoch
        save_model(model)
        logger.info(f"Checkpoint saved for epoch {epoch + 1}")

    return train_losses

@track_time
def evaluate_model(model, eval_dataloader, column_names, dataset):
    """Evaluates model on validation data, filtering and deduplicating outputs."""
    logger.info("Evaluating model...")
    model.eval()
    unique_pairs = {}
    total_loss = 0
    total_batches = len(eval_dataloader)

    with open(config["eval_output_file"], "w", encoding="utf-8") as f:
        for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):  # Added progress bar
            batch = {k: v.to(device) for k, v in batch.items()}
            original_sql = dataset["validation"][i]["sql"]
            safe_query = sql_dict_to_string(original_sql, column_names)

            with torch.no_grad():
                outputs = model(**batch)
                total_loss += outputs.loss.item()

                generated_ids = model.generate(
                    batch["input_ids"],
                    max_length=config["max_length"],
                    num_beams=config["beam_size"],
                    repetition_penalty=config["repetition_penalty"],
                    temperature=config["temperature"],
                    early_stopping=config["early_stopping"],
                    do_sample=False
                )
                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred in preds:
                pair_key = (safe_query, pred.strip().lower())
                if pair_key not in unique_pairs:
                    unique_pairs[pair_key] = True
                    f.write(f"SQL: {safe_query}\nGenerated Question: {pred}\n\n")

            if i % (total_batches // 5) == 0:
                logger.info(f"Evaluation progress: {i * 100 / total_batches:.0f}%")

    avg_val_loss = total_loss / total_batches
    logger.info("Evaluation complete.")
    return avg_val_loss

# Set up logging
def setup_logging(config):
    """Set up logging to both file and console"""
    # Create logs directory if it doesn't exist
    os.makedirs(config["logs_dir"], exist_ok=True)

    # Create timestamp for log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config["logs_dir"], f"training_log_{timestamp}.txt")

    # Set up logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(config["logging_level"])
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial info
    logger.info(f"Log file created at: {log_file}")
    logger.info(f"Running with configuration: {config['run_name']}")

    return logger

# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Set up logging
    logger = setup_logging(config)

    device = torch.device("cuda" if config["gpu_available"] else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer, model = load_or_train_model_and_tokenizer()
    tokenized_dataset, column_names = load_and_preprocess_dataset(tokenizer)
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=config["batch_size"], collate_fn=collate_fn)
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=config["batch_size"], collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=len(train_dataloader) * config["epochs"])

    # Train model
    train_losses = train_model(model, optimizer, scheduler, train_dataloader)

    # Save model after training
    save_model(model)
    logger.info("Model saved successfully.")

    # Evaluate model
    final_val_loss = evaluate_model(model, eval_dataloader, column_names, tokenized_dataset)

    # Save run data and plot comparisons
    save_run_data(config, train_losses, final_val_loss)
    plot_run_data(config["run_name"], train_losses, final_val_loss)

    logger.info(f"Total script execution time: {time.time() - start_time:.2f} seconds")
