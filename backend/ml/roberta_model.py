import os
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models", "roberta")
RESULT_DIR = os.path.join(
    BASE_DIR, "..", "research", "transformer_results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 55)
print("SENTINELAI - ROBERTA THREAT DETECTION")
print("=" * 55)
print("Device:", DEVICE)


# ==========================================
# LOAD DATASET
# ==========================================

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} dataset contains missing values.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{name} dataset has unexpected labels.")

print("\nDataset loaded successfully.")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))


# ==========================================
# TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


# ==========================================
# CONVERT TO HUGGING FACE DATASET
# ==========================================

def prepare_dataset(df):

    dataset = Dataset.from_pandas(
        df[["text", "label"]],
        preserve_index=False
    )

    dataset = dataset.rename_column("label", "labels")

    return dataset.map(
        tokenize_function,
        batched=True
    )


train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==========================================
# LOAD ROBERTA MODEL
# ==========================================

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


# ==========================================
# EVALUATION METRICS
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, zero_division=0
        )
    }


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "checkpoints"),

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,

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


# ==========================================
# TRAIN MODEL
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

print("\nStarting RoBERTa training...")

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print("\nTraining completed.")
print(f"Training time: {training_time:.2f} seconds")


# ==========================================
# SAVE MODEL
# ==========================================

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("\nModel saved:", MODEL_DIR)


# ==========================================
# VALIDATION RESULTS
# ==========================================

validation_results = trainer.evaluate(
    eval_dataset=val_dataset
)

print("\nVALIDATION RESULTS")
print(validation_results)


# ==========================================
# TEST EVALUATION
# ==========================================

predictions = trainer.predict(test_dataset)

y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

test_results = {
    "model": "RoBERTa",
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_true, y_pred, zero_division=0)
    ),
    "training_time_seconds": float(training_time)
}

print("\nTEST RESULTS")
print(json.dumps(test_results, indent=4))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["BENIGN", "MALICIOUS"],
        zero_division=0
    )
)


# ==========================================
# SAVE RESULTS
# ==========================================

results_path = os.path.join(
    RESULT_DIR,
    "roberta_test_results.json"
)

with open(results_path, "w") as file:
    json.dump(test_results, file, indent=4)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "MALICIOUS"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("SentinelAI - RoBERTa Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "roberta_confusion_matrix.png"),
    dpi=300
)

plt.close()

print("\nResults saved successfully.")
print("=" * 55)
print("ROBERTA EXPERIMENT COMPLETED")
print("=" * 55)
