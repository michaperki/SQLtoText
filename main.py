
import torch
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm  # for progress bar

# Configuration Settings
config = {
    "model_name": "t5-small",
    "batch_size": 8,
    "max_length": 128,  # Reduced sequence length for now, adjust later
    "learning_rate": 5e-5,
    "epochs": 3,
    "gradient_accumulation_steps": 4,
    "use_amp": False,
    "debug": False,
    "logging_level": "DEBUG" if True else "INFO",
    "gpu_available": torch.cuda.is_available(),
    "sample_size": 1000,  # Limit dataset size for faster testing
}

# Set up logging
logging.basicConfig(level=config["logging_level"],
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Step 1: Load the model and tokenizer
logger.info("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

# Move model to GPU if available
device = torch.device("cuda" if config["gpu_available"] else "cpu")
model.to(device)
logger.info(f"Using device: {device}")

# Step 2: Load the dataset
logger.info("Loading dataset...")
dataset = load_dataset("wikisql")

# Limit dataset to a subset for faster training
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(config["sample_size"]))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(config["sample_size"]))

# Step 4: SQL dictionary to string conversion function
def sql_dict_to_string(sql_dict):
    try:
        select_clause = f'SELECT {", ".join(sql_dict["select"][1])}'
    except KeyError:
        select_clause = "SELECT *"  # Fallback if 'select' key doesn't exist

    where_clause = ""
    if "conds" in sql_dict and len(sql_dict['conds']) > 0:
        conditions = []
        for col, op, cond in zip(sql_dict['conds']['column_index'], sql_dict['conds']['operator_index'], sql_dict['conds']['condition']):
            conditions.append(f'Column_{col} {op} "{cond}"')
        where_clause = " AND ".join(conditions)

    query = select_clause
    if where_clause:
        query += f' WHERE {where_clause}'
    return query

# Step 5: Preprocess the dataset to prepare inputs for the model
def preprocess_function(examples):
    inputs = ["translate SQL to English: " + sql_dict_to_string(item) for item in examples["sql"]]
    targets = [item for item in examples["question"]]

    model_inputs = tokenizer(inputs, max_length=config["max_length"], truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, max_length=config["max_length"], truncation=True, padding="max_length", return_tensors="pt").input_ids

    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

# Step 6: Tokenize the dataset
logger.info("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["sql", "question"])

# Step 7: Define a custom collate function for padding consistency
def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) if isinstance(item['input_ids'], list) else item['input_ids'] for item in batch]
    labels = [torch.tensor(item['labels']) if isinstance(item['labels'], list) else item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {'input_ids': input_ids, 'labels': labels}

# Step 8: Create DataLoader for batching with the custom collate function
from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=config["batch_size"], collate_fn=collate_fn)
eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=config["batch_size"], collate_fn=collate_fn)

# Step 9: Set up training
logger.info("Setting up optimizer and training configurations...")
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

# Training loop with progress bar
for epoch in range(config["epochs"]):
    logger.info(f"Starting epoch {epoch + 1}/{config['epochs']}...")
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0

    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss / config["gradient_accumulation_steps"]
        loss.backward()

        # Optimizer step after accumulating gradients
        if (i + 1) % config["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

    logger.info(f"Epoch {epoch + 1} completed with avg loss: {epoch_loss / len(train_dataloader)}")

# Step 10: Evaluate the model
logger.info("Evaluating model...")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        generated_ids = model.generate(batch["input_ids"], max_length=config["max_length"])
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for pred in preds:
        print(pred)

