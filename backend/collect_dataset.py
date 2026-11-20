from datasets import load_dataset
import pandas as pd
import os

print("===================================")
print("       SENTINELAI DATA COLLECTION")
print("===================================")

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load dataset
print("\nLoading Prompt Injection Dataset...")

dataset = load_dataset(
    "S-Labs/prompt-injection-dataset"
)

# Display available splits
print("\nAvailable Dataset Splits:")
print(dataset)

# Convert each split into CSV
for split in dataset:

    df = dataset[split].to_pandas()

    output_path = f"data/raw/{split}.csv"

    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")

    print("Shape:", df.shape)

    print("\nSample Records:")
    print(df.head())

print("\n===================================")
print("DATASET COLLECTION COMPLETED")
print("===================================")