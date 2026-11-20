# ==================================================
# SENTINELAI - DISTILBERT THREAT DETECTION
# ==================================================

import os
import time
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. DIRECTORY CONFIGURATION
# ==================================================

ML_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(ML_DIR)

PROJECT_DIR = os.path.dirname(BACKEND_DIR)

RAW_DIR = os.path.join(
    BACKEND_DIR, "data", "raw"
)

MODEL_DIR = os.path.join(
    BACKEND_DIR, "models", "distilbert"
)

RESEARCH_DIR = os.path.join(
    PROJECT_DIR, "research", "transformer_results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESEARCH_DIR, exist_ok=True)


print("\n==========================================")
print(" SENTINELAI - DISTILBERT TRAINING")
print("==========================================")


# ==================================================
# 2. DEVICE CONFIGURATION
# ==================================================

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nComputing Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU. Training may take longer.")


# ==================================================
# 3. LOAD DATASETS
# ==================================================

train_df = pd.read_csv(
    os.path.join(RAW_DIR, "train.csv")
)

val_df = pd.read_csv(
    os.path.join(RAW_DIR, "validation.csv")
)

test_df = pd.read_csv(
    os.path.join(RAW_DIR, "test.csv")
)


print("\nDataset Information")

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)


# ==================================================
# 4. DATA VALIDATION
# ==================================================

for name, df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    if df[["text", "label"]].isnull().any().any():
        raise ValueError(
            f"Missing values found in {name} dataset"
        )

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels in {name} dataset"
        )


# ==================================================
# 5. PREPARE HUGGING FACE DATASETS
# ==================================================

train_data = Dataset.from_pandas(
    train_df[["text", "label"]].rename(
        columns={"label": "labels"}
    ),
    preserve_index=False
)

val_data = Dataset.from_pandas(
    val_df[["text", "label"]].rename(
        columns={"label": "labels"}
    ),
    preserve_index=False
)

test_data = Dataset.from_pandas(
    test_df[["text", "label"]].rename(
        columns={"label": "labels"}
    ),
    preserve_index=False
)


# ==================================================
# 6. LOAD DISTILBERT TOKENIZER
# ==================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

print("\nLoading DistilBERT Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==================================================
# 7. TOKENIZATION
# ==================================================

MAX_LENGTH = 256

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("\nTokenizing datasets...")

train_data = train_data.map(
    tokenize_function,
    batched=True
)

val_data = val_data.map(
    tokenize_function,
    batched=True
)

test_data = test_data.map(
    tokenize_function,
    batched=True
)


# ==================================================
# 8. DATA COLLATOR
# ==================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==================================================
# 9. LOAD DISTILBERT MODEL
# ==================================================

print("\nLoading pretrained DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=2,

    id2label={
        0: "BENIGN",
        1: "MALICIOUS"
    },

    label2id={
        "BENIGN": 0,
        "MALICIOUS": 1
    }

)


# ==================================================
# 10. EVALUATION METRICS
# ==================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
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


# ==================================================
# 11. TRAINING CONFIGURATION
# ==================================================

training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "checkpoints"),

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",

    save_total_limit=2,

    logging_steps=100,

    report_to="none",

    fp16=torch.cuda.is_available()
)


# ==================================================
# 12. INITIALIZE TRAINER
# ==================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_data,

    eval_dataset=val_data,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics

)


# ==================================================
# 13. TRAIN MODEL
# ==================================================

print("\n==========================================")
print(" STARTING DISTILBERT TRAINING")
print("==========================================")

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print(
    "\nTraining Time:",
    round(training_time, 2),
    "seconds"
)


# ==================================================
# 14. SAVE TRAINED MODEL
# ==================================================

print("\nSaving DistilBERT model...")

trainer.save_model(MODEL_DIR)

tokenizer.save_pretrained(MODEL_DIR)

print("Model saved successfully!")


# ==================================================
# 15. VALIDATION EVALUATION
# ==================================================

print("\n========== VALIDATION RESULTS ==========")

validation_results = trainer.evaluate(
    val_data
)

print(validation_results)


# ==================================================
# 16. FINAL TEST EVALUATION
# ==================================================

print("\n========== FINAL TEST EVALUATION ==========")

test_output = trainer.predict(
    test_data
)

test_predictions = np.argmax(
    test_output.predictions,
    axis=1
)

test_labels = test_output.label_ids

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)


print("\nAccuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)


print("\nClassification Report")

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["Benign", "Malicious"],
        zero_division=0
    )
)


# ==================================================
# 17. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",

    xticklabels=[
        "Benign",
        "Malicious"
    ],

    yticklabels=[
        "Benign",
        "Malicious"
    ]
)

plt.title(
    "SentinelAI - DistilBERT Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESEARCH_DIR,
        "distilbert_confusion_matrix.png"
    )
)

plt.close()


# ==================================================
# 18. SAVE RESULTS
# ==================================================

results = {

    "Model": "DistilBERT",

    "Accuracy": float(accuracy),

    "Precision": float(precision),

    "Recall": float(recall),

    "F1 Score": float(f1),

    "Training Time": float(training_time)

}


with open(
    os.path.join(
        RESEARCH_DIR,
        "distilbert_results.json"
    ),
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\n==========================================")
print(" DISTILBERT TRAINING COMPLETED")
print("==========================================")
