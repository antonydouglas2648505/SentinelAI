import torch
import pandas as pd

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "roberta"

RESULT_DIR = BASE_DIR.parent / "research" / "augmentation_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
MAX_LENGTH = 256

THRESHOLDS = [0.20, 0.50]


def main():

    print("=" * 65)
    print(" SENTINELAI - SELECTED THRESHOLD TEST EVALUATION")
    print("=" * 65)

    df = pd.read_csv(TEST_PATH)
    df = df.dropna(subset=["text", "label"])

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    print(f"\nTest samples: {len(df)}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    model.to(DEVICE)
    model.eval()

    probabilities_list = []

    print("\nGenerating predictions...")

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

        probabilities_list.extend(
            probabilities[:, 1].cpu().numpy()
        )

    results = []

    for threshold in THRESHOLDS:

        predictions = [
            1 if probability >= threshold else 0
            for probability in probabilities_list
        ]

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        ).ravel()

        result = {
            "model": "Original RoBERTa",
            "threshold": threshold,
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(
                labels, predictions, zero_division=0
            ),
            "recall": recall_score(
                labels, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                labels, predictions, zero_division=0
            ),
            "false_positive_rate": fp / (fp + tn),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }

        results.append(result)

    results_df = pd.DataFrame(results)

    output_path = RESULT_DIR / "selected_threshold_test_results.csv"

    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print(" FINAL TEST DATASET COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()