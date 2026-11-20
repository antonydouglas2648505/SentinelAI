import pandas as pd
import numpy as np
import os
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================

ML_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(ML_DIR)

RAW_DIR = os.path.join(
    BACKEND_DIR, "data", "raw"
)

PROCESSED_DIR = os.path.join(
    BACKEND_DIR, "data", "processed"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("\n======================================")
print(" SENTINELAI DATA PREPROCESSING")
print("======================================")

# ==========================================
# 2. LOAD DATASETS
# ==========================================

train = pd.read_csv(
    os.path.join(RAW_DIR, "train.csv")
)

validation = pd.read_csv(
    os.path.join(RAW_DIR, "validation.csv")
)

test = pd.read_csv(
    os.path.join(RAW_DIR, "test.csv")
)

datasets = {
    "train": train,
    "validation": validation,
    "test": test
}

print("\nDatasets Loaded Successfully")

# ==========================================
# 3. DATA VALIDATION
# ==========================================

for name, df in datasets.items():

    print(f"\n{name.upper()} DATASET")

    print("Original Shape:", df.shape)

    required_columns = ["text", "label"]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Missing required column: {col}"
            )

    print("Required columns verified.")

# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

for name, df in datasets.items():

    print(f"\nProcessing {name}")

    df.dropna(
        subset=["text", "label"],
        inplace=True
    )

    df["text"] = df["text"].astype(str)

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise"
    ).astype(int)

    print("Missing values handled.")

# ==========================================
# 5. REMOVE EMPTY TEXT
# ==========================================

for name, df in datasets.items():

    before = len(df)

    df.drop(
        index=df[
            df["text"].str.strip().eq("")
        ].index,
        inplace=True
    )

    after = len(df)

    print(
        f"{name}: Removed {before - after} empty records"
    )

# ==========================================
# 6. VALIDATE LABELS
# ==========================================

for name, df in datasets.items():

    invalid_labels = set(
        df["label"].unique()
    ) - {0, 1}

    if invalid_labels:

        raise ValueError(
            f"Unexpected labels in {name}: {invalid_labels}"
        )

    print(f"{name}: Labels verified.")

# ==========================================
# 7. DUPLICATE ANALYSIS
# ==========================================

print("\n========== DUPLICATE ANALYSIS ==========")

for name, df in datasets.items():

    duplicates = df.duplicated(
        subset=["text"]
    ).sum()

    print(
        f"{name}: {duplicates} duplicate prompts"
    )

# ==========================================
# 8. TEXT STATISTICS
# ==========================================

for name, df in datasets.items():

    df["character_count"] = (
        df["text"].str.len()
    )

    df["word_count"] = (
        df["text"].str.split().str.len()
    )

    print(f"\n{name.upper()} STATISTICS")

    print(
        df[
            ["character_count", "word_count"]
        ].describe()
    )

# ==========================================
# 9. SAVE PROCESSED DATASETS
# ==========================================

for name, df in datasets.items():

    output_path = os.path.join(
        PROCESSED_DIR,
        f"{name}_processed.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

# ==========================================
# 10. TF-IDF FEATURE EXTRACTION
# ==========================================

print("\n========== TF-IDF FEATURE EXTRACTION ==========")

# Word-level TF-IDF
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b"
)

# Character-level TF-IDF
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    max_features=10000,
    ngram_range=(3, 5),
    sublinear_tf=True
)

# Fit ONLY on training data
X_train_word = word_vectorizer.fit_transform(
    train["text"]
)

X_validation_word = word_vectorizer.transform(
    validation["text"]
)

X_test_word = word_vectorizer.transform(
    test["text"]
)

X_train_char = char_vectorizer.fit_transform(
    train["text"]
)

X_validation_char = char_vectorizer.transform(
    validation["text"]
)

X_test_char = char_vectorizer.transform(
    test["text"]
)

# Combine word and character features
from scipy.sparse import hstack

X_train = hstack([
    X_train_word,
    X_train_char
]).tocsr()

X_validation = hstack([
    X_validation_word,
    X_validation_char
]).tocsr()

X_test = hstack([
    X_test_word,
    X_test_char
]).tocsr()

# ==========================================
# 11. SAVE FEATURE MATRICES
# ==========================================

save_npz(
    os.path.join(PROCESSED_DIR, "X_train.npz"),
    X_train
)

save_npz(
    os.path.join(PROCESSED_DIR, "X_validation.npz"),
    X_validation
)

save_npz(
    os.path.join(PROCESSED_DIR, "X_test.npz"),
    X_test
)

# Save labels
np.save(
    os.path.join(PROCESSED_DIR, "y_train.npy"),
    train["label"].values
)

np.save(
    os.path.join(PROCESSED_DIR, "y_validation.npy"),
    validation["label"].values
)

np.save(
    os.path.join(PROCESSED_DIR, "y_test.npy"),
    test["label"].values
)

# Save vectorizers
import joblib

joblib.dump(
    word_vectorizer,
    os.path.join(PROCESSED_DIR, "word_vectorizer.pkl")
)

joblib.dump(
    char_vectorizer,
    os.path.join(PROCESSED_DIR, "char_vectorizer.pkl")
)

# ==========================================
# 12. FEATURE SUMMARY
# ==========================================

summary = {
    "training_samples": int(X_train.shape[0]),

    "validation_samples": int(X_validation.shape[0]),

    "testing_samples": int(X_test.shape[0]),

    "word_features": int(X_train_word.shape[1]),

    "character_features": int(X_train_char.shape[1]),

    "total_features": int(X_train.shape[1])
}

with open(
    os.path.join(PROCESSED_DIR, "feature_summary.json"),
    "w"
) as file:

    json.dump(summary, file, indent=4)

print("\n========== FEATURE SUMMARY ==========")

print(json.dumps(summary, indent=4))

print("\n======================================")
print(" PREPROCESSING COMPLETED SUCCESSFULLY")
print("======================================")