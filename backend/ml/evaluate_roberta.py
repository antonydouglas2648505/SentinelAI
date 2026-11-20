import json
import sys
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# SENTINELAI - ROBERTA EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "evaluation"
    / "prompt_injection_test.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "research"
    / "evaluation_results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.detector import detector


def normalize_probability(value):
    """
    Convert probabilities to a 0-1 range.
    Supports either decimal or percentage values.
    """

    value = float(value)

    if value > 1:
        value = value / 100

    return max(0.0, min(1.0, value))


def main():

    print("\n" + "=" * 60)
    print(" SENTINELAI - ROBERTA EVALUATION")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"text", "label", "category"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain: {required_columns}"
        )

    if df.empty:
        raise ValueError("Evaluation dataset is empty.")

    y_true = []
    y_pred = []
    probabilities = []

    print(f"\nLoaded samples: {len(df)}")
    print("\nRunning RoBERTa predictions...\n")

    for index, row in df.iterrows():

        text = str(row["text"])

        result = detector.predict(text)

        malicious_probability = normalize_probability(
            result["malicious_probability"]
        )

        prediction = int(malicious_probability >= 0.5)

        y_true.append(int(row["label"]))
        y_pred.append(prediction)
        probabilities.append(malicious_probability)

        status = "CORRECT" if prediction == int(row["label"]) else "INCORRECT"

        print(
            f"[{index + 1}/{len(df)}] "
            f"Expected: {row['label']} | "
            f"Predicted: {prediction} | "
            f"Malicious probability: "
            f"{malicious_probability:.2%} | "
            f"{status}"
        )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print("\n" + "=" * 60)
    print(" OVERALL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1-score:           {f1:.4f}")
    print(f"False positive rate:{false_positive_rate:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            zero_division=0
        )
    )

    # ========================================================
    # CATEGORY PERFORMANCE
    # ========================================================

    print("\n" + "=" * 60)
    print(" CATEGORY PERFORMANCE")
    print("=" * 60)

    evaluation_df = df.copy()

    evaluation_df["predicted_label"] = y_pred
    evaluation_df["malicious_probability"] = probabilities

    category_results = []

    for category, group in evaluation_df.groupby("category"):

        category_accuracy = accuracy_score(
            group["label"],
            group["predicted_label"]
        )

        category_results.append({
            "category": category,
            "samples": len(group),
            "accuracy": category_accuracy,
            "detected_malicious": int(
                (
                    (group["label"] == 1)
                    & (group["predicted_label"] == 1)
                ).sum()
            )
        })

        print(
            f"{category}: "
            f"{category_accuracy:.2%} accuracy "
            f"({len(group)} samples)"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "model": "RoBERTa",
        "evaluation_dataset": DATASET_PATH.name,
        "total_samples": len(df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }

    results_path = RESULTS_DIR / "roberta_evaluation.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    category_path = RESULTS_DIR / "roberta_category_results.csv"

    pd.DataFrame(category_results).to_csv(
        category_path,
        index=False
    )

    detailed_path = RESULTS_DIR / "roberta_predictions.csv"

    evaluation_df.to_csv(
        detailed_path,
        index=False
    )

    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETED")
    print("=" * 60)

    print(f"Overall results: {results_path}")
    print(f"Category results: {category_path}")
    print(f"Individual predictions: {detailed_path}")


if __name__ == "__main__":
    main()