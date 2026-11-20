# ==========================================
# SENTINELAI - EXPLORATORY DATA ANALYSIS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import Counter

# ==========================================
# 1. INITIAL CONFIGURATION
# ==========================================
# Get project directories
ML_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(ML_DIR)

PROJECT_DIR = os.path.dirname(BACKEND_DIR)

# Dataset paths
train_path = os.path.join(
    BACKEND_DIR, "data", "raw", "train.csv"
)

val_path = os.path.join(
    BACKEND_DIR, "data", "raw", "validation.csv"
)

test_path = os.path.join(
    BACKEND_DIR, "data", "raw", "test.csv"
)

# Output directory
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "research", "dataset_analysis"
)

# Automatically create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nOutput directory:", OUTPUT_DIR)

# ==========================================
# 2. LOAD DATASETS
# ==========================================

train_path = "data/raw/train.csv"
val_path = "data/raw/validation.csv"
test_path = "data/raw/test.csv"

train = pd.read_csv(train_path)
validation = pd.read_csv(val_path)
test = pd.read_csv(test_path)

print("\nDATASET LOADED SUCCESSFULLY")

print("\nTraining Shape:", train.shape)
print("Validation Shape:", validation.shape)
print("Testing Shape:", test.shape)

# ==========================================
# 3. COMBINE FOR EXPLORATORY ANALYSIS
# ==========================================

train["split"] = "Train"
validation["split"] = "Validation"
test["split"] = "Test"

df = pd.concat(
    [train, validation, test],
    ignore_index=True
)

print("\nCombined Dataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nFirst Five Records:")
print(df.head())

# ==========================================
# 4. MISSING VALUE ANALYSIS
# ==========================================

print("\nMISSING VALUE ANALYSIS")

print(df.isnull().sum())

plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Value Analysis")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/missing_values.png"
)

plt.close()

# ==========================================
# 5. DUPLICATE ANALYSIS
# ==========================================

print("\nDUPLICATE ANALYSIS")

duplicates = df.duplicated(subset=["text"]).sum()

print("Duplicate Prompts:", duplicates)

# ==========================================
# 6. CLASS DISTRIBUTION
# ==========================================

print("\nCLASS DISTRIBUTION")

print(df["label"].value_counts())

class_counts = df["label"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=class_counts.index,
    y=class_counts.values
)

plt.xticks(
    [0, 1],
    ["Benign", "Malicious"]
)

plt.title("Benign vs Malicious Prompt Distribution")

plt.xlabel("Prompt Category")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/class_distribution.png"
)

plt.close()

# ==========================================
# 7. CLASS DISTRIBUTION BY SPLIT
# ==========================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="split",
    hue="label"
)

plt.title("Class Distribution Across Dataset Splits")

plt.xlabel("Dataset Split")
plt.ylabel("Number of Samples")

plt.legend(
    title="Class",
    labels=["Benign", "Malicious"]
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/split_distribution.png"
)

plt.close()

# ==========================================
# 8. TEXT LENGTH ANALYSIS
# ==========================================

df["text_length"] = df["text"].astype(str).apply(len)

df["word_count"] = df["text"].astype(str).apply(
    lambda x: len(x.split())
)

print("\nTEXT STATISTICS")

print(df["text_length"].describe())

print("\nWORD COUNT STATISTICS")

print(df["word_count"].describe())

# ==========================================
# 9. TEXT LENGTH DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="text_length",
    hue="label",
    bins=40,
    kde=True
)

plt.title("Prompt Length Distribution")

plt.xlabel("Number of Characters")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/text_length_distribution.png"
)

plt.close()

# ==========================================
# 10. WORD COUNT DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="label",
    y="word_count"
)

plt.xticks(
    [0, 1],
    ["Benign", "Malicious"]
)

plt.title("Word Count Comparison")

plt.xlabel("Prompt Category")
plt.ylabel("Word Count")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/word_count_distribution.png"
)

plt.close()

# ==========================================
# 11. MOST FREQUENT WORDS
# ==========================================

def extract_words(text):

    text = str(text).lower()

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)

    return words


benign_text = " ".join(
    df[df["label"] == 0]["text"].astype(str)
)

malicious_text = " ".join(
    df[df["label"] == 1]["text"].astype(str)
)

benign_words = Counter(
    extract_words(benign_text)
)

malicious_words = Counter(
    extract_words(malicious_text)
)

print("\nTOP BENIGN WORDS")

print(benign_words.most_common(20))

print("\nTOP MALICIOUS WORDS")

print(malicious_words.most_common(20))


def plot_top_words(counter, title, filename):

    words = counter.most_common(15)

    word_df = pd.DataFrame(
        words,
        columns=["Word", "Frequency"]
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=word_df,
        x="Frequency",
        y="Word"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}"
    )

    plt.close()


plot_top_words(
    benign_words,
    "Most Frequent Benign Words",
    "benign_word_frequency.png"
)

plot_top_words(
    malicious_words,
    "Most Frequent Malicious Words",
    "malicious_word_frequency.png"
)

# ==========================================
# 12. CLASS-WISE STATISTICS
# ==========================================

class_statistics = df.groupby("label").agg(
    total_samples=("text", "count"),
    average_length=("text_length", "mean"),
    average_words=("word_count", "mean"),
    minimum_length=("text_length", "min"),
    maximum_length=("text_length", "max")
)

print("\nCLASS-WISE STATISTICS")

print(class_statistics)

class_statistics.to_csv(
    f"{OUTPUT_DIR}/class_statistics.csv"
)

# ==========================================
# 13. DATASET SUMMARY
# ==========================================

summary = pd.DataFrame({
    "Metric": [
        "Total Samples",
        "Benign Samples",
        "Malicious Samples",
        "Duplicate Prompts",
        "Missing Values",
        "Average Text Length",
        "Average Word Count"
    ],

    "Value": [
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
        duplicates,
        df.isnull().sum().sum(),
        df["text_length"].mean(),
        df["word_count"].mean()
    ]
})

print("\nFINAL DATASET SUMMARY")

print(summary)

summary.to_csv(
    f"{OUTPUT_DIR}/eda_summary.csv",
    index=False
)

print("\n======================================")
print(" SENTINELAI EDA COMPLETED")
print("======================================")