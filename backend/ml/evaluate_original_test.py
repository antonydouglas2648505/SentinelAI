import json
import torch
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"

MODELS = {
    "Original RoBERTa": BASE_DIR / "models" / "roberta",
    "Augmented RoBERTa": BASE_DIR / "models" / "roberta_augmented"
}

RESULT_DIR = BASE_DIR.parent / "research" / "augmentation_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LENGTH = 256


def evaluate_model(model_name, model_path, texts, labels):

    print(f"\nEvaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    model.to(DEVICE)
    model.eval()

    predictions = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        batch_predictions = torch.argmax(
            probabilities,
            dim=1
        ).cpu().numpy()

        predictions.extend(batch_predictions)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


def main():

    print("=" * 65)
    print(" SENTINELAI - ORIGINAL TEST DATASET EVALUATION")
    print("=" * 65)

    df = pd.read_csv(TEST_PATH)

    df = df.dropna(subset=["text", "label"])

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    print(f"\nTest samples: {len(df)}")
    print(f"Benign samples: {labels.count(0)}")
    print(f"Malicious samples: {labels.count(1)}")
    print(f"Device: {DEVICE}")

    results = []

    for model_name, model_path in MODELS.items():

        result = evaluate_model(
            model_name,
            model_path,
            texts,
            labels
        )

        results.append(result)

        print(f"\n{model_name} Results")
        print("-" * 40)

        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    results_df = pd.DataFrame(results)

    output_path = RESULT_DIR / "original_test_model_comparison.csv"

    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print(" FINAL MODEL COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()