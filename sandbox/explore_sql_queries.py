
import os
from datasets import load_dataset
import json

# Adjust this if you want to use a cached dataset or specific sample size
sample_size = 10
cache_dir = "./cache"

# Load dataset
def load_sample_dataset():
    os.makedirs(cache_dir, exist_ok=True)
    print("Loading dataset...")
    dataset = load_dataset("wikisql", cache_dir=cache_dir)

    # Select a sample from the training set for exploration
    sample_data = dataset['train'].shuffle(seed=42).select(range(sample_size))
    return sample_data

# Print sample SQL queries from the dataset
def explore_sql_queries(sample_data):
    for i, item in enumerate(sample_data):
        sql_query = item.get("sql", {})
        question = item.get("question", "N/A")
        table_header = item.get("table", {}).get("header", [])

        print(f"--- Sample {i + 1} ---")
        print("Original Question:", question)
        print("SQL Dictionary:", json.dumps(sql_query, indent=4))
        print("Table Headers:", table_header)
        print("\n")

if __name__ == "__main__":
    # Load a sample of the dataset
    sample_data = load_sample_dataset()

    # Explore and print SQL queries and questions
    explore_sql_queries(sample_data)
