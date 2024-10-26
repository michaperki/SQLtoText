"""
Data handling module for loading, preprocessing, and managing datasets.
"""

import os
import logging
from typing import Tuple, Dict, List
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class SQLDataProcessor:
    """Handles loading, preprocessing and managing SQL-to-question datasets."""

    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        """
        Initialize the data processor.

        Args:
            config: Configuration object containing data settings
            tokenizer: Tokenizer for processing text
        """
        self.config = config
        self.tokenizer = tokenizer
        self.column_names = None

    def sql_dict_to_string(self, sql_dict: Dict, column_names: List[str]) -> str:
        """
        Convert a SQL dictionary to a string representation.

        Args:
            sql_dict: Dictionary containing SQL query components
            column_names: List of column names for the table

        Returns:
            str: String representation of the SQL query
        """
        try:
            select_clause = (f'SELECT {column_names[sql_dict["select"][1]]}'
                           if sql_dict["select"][1] < len(column_names)
                           else "SELECT *")
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

    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess a batch of examples.

        Args:
            examples: Dictionary containing batch of examples

        Returns:
            Dict: Processed examples with input_ids and labels
        """
        inputs = [
            f"Rephrase this SQL query into a natural language question: {self.sql_dict_to_string(item, self.column_names)}"
            for item in examples["sql"]
        ]
        targets = examples["question"]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = self.tokenizer(
            targets,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids

        # Replace padding token id with -100 for training
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        # Log sample for verification
        logger.info(f"SQL: {inputs[0]}")
        logger.info(f"Generated Question: {targets[0]}")

        return model_inputs

    @staticmethod
    def collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict:
        """
        Collate a batch of examples.

        Args:
            batch: List of examples to collate
            tokenizer: Tokenizer for handling padding

        Returns:
            Dict: Collated batch with input_ids and labels
        """
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True,
                               padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {'input_ids': input_ids, 'labels': labels}

    def load_and_process_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load, preprocess and create data loaders for the dataset.

        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation data loaders
        """
        # Create cache directory if it doesn't exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
        logger.info("Loading dataset...")

        # Load dataset
        dataset = load_dataset("wikisql")
        logger.info(f"Dataset loaded with {len(dataset['train'])} training samples "
                   f"and {len(dataset['validation'])} validation samples")

        # Setup cache file
        cache_file = os.path.join(
            self.config.cache_dir,
            f"tokenized_dataset_{self.config.sample_size}.arrow"
        )

        # Load from cache if available
        if os.path.exists(cache_file):
            logger.info("Loading tokenized dataset from cache...")
            tokenized_dataset = DatasetDict.load_from_disk(cache_file)
        else:
            # Sample and preprocess dataset
            if self.config.sample_size > 0:  # Only sample if sample_size is positive
                dataset["train"] = dataset["train"].shuffle(seed=42).select(
                    range(self.config.sample_size)
                )
                dataset["validation"] = dataset["validation"].shuffle(seed=42).select(
                    range(min(100, len(dataset["validation"])))
                )

            # Store column names for SQL processing
            self.column_names = dataset['train'][0]['table']['header']
            logger.info(f"Sample column names: {self.column_names[:5]}... (truncated)")

            # Preprocess dataset
            tokenized_dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=["question"]
            )

            # Cache processed dataset
            tokenized_dataset.save_to_disk(cache_file)
            logger.info("Tokenized dataset cached successfully")

        # Create data loaders
        train_dataloader = DataLoader(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=self.config.batch_size,
            collate_fn=lambda batch: self.collate_fn(batch, self.tokenizer)
        )

        eval_dataloader = DataLoader(
            tokenized_dataset["validation"],
            batch_size=self.config.batch_size,
            collate_fn=lambda batch: self.collate_fn(batch, self.tokenizer)
        )

        return train_dataloader, eval_dataloader, self.column_names

    def get_sample_batch(self) -> Dict:
        """
        Get a sample batch for testing purposes.

        Returns:
            Dict: Sample batch with input_ids and labels
        """
        test_sql_query = "SELECT Points FROM table WHERE Tries = 55"
        prompt = f"Rephrase this SQL query into a natural language question: {test_sql_query}"

        return self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length"
        )
