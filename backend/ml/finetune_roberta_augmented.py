import os
import json
import time
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

AUGMENTATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "augmentation",
    "hard_negative_training.csv"
)

BASELINE_MODEL_DIR = os.path.join(
    BASE_DIR, "models", "roberta"
)

OUTPUT_MODEL_DIR = os.path.join(
    BASE_DIR, "models", "roberta_augmented"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "research",
    "augmentation_results"
)

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_LENGTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("=" * 65)
print(" SENTINELAI - AUGMENTED ROBERTA FINE-TUNING")
print("=" * 65)

print("Device:", DEVICE)

if not os.path.isdir(BASELINE_MODEL_DIR):
    raise FileNotFoundError(
        f"Baseline model not found: {BASELINE_MODEL_DIR}"
    )

if not os.path.isfile(AUGMENTATION_FILE):
    raise FileNotFoundError(
        f"Augmentation dataset not found: {AUGMENTATION_FILE}"
    )


# ==========================================
# LOAD DATASETS
# ==========================================

train_df = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv")
)

val_df = pd.read_csv(
    os.path.join(DATA_DIR, "validation.csv")
)

augmentation_df = pd.read_csv(AUGMENTATION_FILE)


def validate_dataset(df, name):

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{name} dataset must contain text and label columns."
        )

    if df["text"].isna().any():
        raise ValueError(
            f"{name} contains missing text values."
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise"
    ).astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"{name} contains invalid labels."
        )

    return df


train_df = validate_dataset(train_df, "Training")
val_df = validate_dataset(val_df, "Validation")
augmentation_df = validate_dataset(
    augmentation_df,
    "Augmentation"
)


# ==========================================
# CHECK DATASET OVERLAP
# ==========================================

def normalize(text):
    return " ".join(
        str(text).lower().split()
    )


augmentation_texts = set(
    augmentation_df["text"].map(normalize)
)

validation_texts = set(
    val_df["text"].map(normalize)
)

overlap = augmentation_texts.intersection(
    validation_texts
)

if overlap:
    raise ValueError(
        f"Augmentation/validation overlap detected: {len(overlap)}"
    )


# ==========================================
# COMBINE TRAINING DATA
# ==========================================

original_train_size = len(train_df)

combined_train_df = pd.concat(
    [
        train_df[["text", "label"]],
        augmentation_df[["text", "label"]]
    ],
    ignore_index=True
)

print("\nDATASET INFORMATION")

print("Original training samples:", original_train_size)

print("Augmentation samples:", len(augmentation_df))

print("Combined training samples:", len(combined_train_df))

print("Validation samples:", len(val_df))

print("\nCombined label distribution:")
print(combined_train_df["label"].value_counts())


# ==========================================
# TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    BASELINE_MODEL_DIR
)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


# ==========================================
# PREPARE DATASETS
# ==========================================

def prepare_dataset(df):

    dataset = Dataset.from_pandas(
        df[["text", "label"]],
        preserve_index=False
    )

    dataset = dataset.rename_column(
        "label",
        "labels"
    )

    return dataset.map(
        tokenize_function,
        batched=True
    )


train_dataset = prepare_dataset(
    combined_train_df
)

val_dataset = prepare_dataset(
    val_df
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==========================================
# LOAD EXISTING TRAINED ROBERTA
# ==========================================

print("\nLoading existing trained RoBERTa...")

model = AutoModelForSequenceClassification.from_pretrained(
    BASELINE_MODEL_DIR
)

print("Baseline model loaded successfully!")


# ==========================================
# EVALUATION METRICS
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        )
    }


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

training_args = TrainingArguments(

    output_dir=os.path.join(
        OUTPUT_MODEL_DIR,
        "checkpoints"
    ),

    learning_rate=1e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=8,

    num_train_epochs=1,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=2,

    logging_steps=100,

    report_to="none",

    fp16=torch.cuda.is_available()
)


# ==========================================
# TRAINER
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ==========================================
# FINE-TUNE MODEL
# ==========================================

print("\nStarting augmented RoBERTa fine-tuning...")

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print("\nFine-tuning completed!")

print(
    f"Training time: {training_time:.2f} seconds"
)


# ==========================================
# SAVE IMPROVED MODEL
# ==========================================

trainer.save_model(
    OUTPUT_MODEL_DIR
)

tokenizer.save_pretrained(
    OUTPUT_MODEL_DIR
)

print("\nImproved model saved:")
print(OUTPUT_MODEL_DIR)


# ==========================================
# VALIDATION EVALUATION
# ==========================================

validation_results = trainer.evaluate()

print("\nVALIDATION RESULTS")

print(validation_results)


# ==========================================
# SAVE EXPERIMENT RESULTS
# ==========================================

results = {

    "model": "RoBERTa Augmented",

    "baseline_model": BASELINE_MODEL_DIR,

    "augmentation_samples": len(augmentation_df),

    "original_training_samples": original_train_size,

    "combined_training_samples": len(combined_train_df),

    "validation_samples": len(val_df),

    "learning_rate": 1e-5,

    "epochs": 1,

    "training_time_seconds": training_time,

    "validation_results": {
        key: float(value)
        for key, value in validation_results.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
}


RESULT_FILE = os.path.join(
    RESULT_DIR,
    "roberta_augmented_results.json"
)

with open(RESULT_FILE, "w") as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ==========================================
# COMPLETION
# ==========================================

print("\n" + "=" * 65)

print(" AUGMENTED ROBERTA EXPERIMENT COMPLETED")

print("=" * 65)

print("\nResults saved:")
print(RESULT_FILE)

print("\nOriginal model preserved:")
print(BASELINE_MODEL_DIR)

print("\nImproved model:")
print(OUTPUT_MODEL_DIR)