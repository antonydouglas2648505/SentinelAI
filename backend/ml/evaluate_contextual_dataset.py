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
# SENTINELAI - CONTEXTUAL DATASET EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "augmentation"
    / "contextual_samples.csv"
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

    value = float(value)

    if value > 1:
        value /= 100

    return max(0.0, min(1.0, value))


def main():

    print("\n" + "=" * 65)
    print(" SENTINELAI - CONTEXTUAL DATASET EVALUATION")
    print("=" * 65)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"text", "label", "category"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["text"].isna().any():
        raise ValueError("Dataset contains missing text values.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    y_true = []
    y_pred = []
    probabilities = []

    print(f"\nTotal samples: {len(df)}")
    print("\nRunning RoBERTa predictions...\n")

    for index, row in df.iterrows():

        result = detector.predict(str(row["text"]))

        probability = normalize_probability(
            result["malicious_probability"]
        )

        prediction = int(probability >= 0.5)

        actual = int(row["label"])

        y_true.append(actual)
        y_pred.append(prediction)
        probabilities.append(probability)

        status = (
            "CORRECT"
            if actual == prediction
            else "INCORRECT"
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"Category: {row['category']} | "
            f"Actual: {actual} | "
            f"Predicted: {prediction} | "
            f"Confidence: {probability:.2%} | "
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

    print("\n" + "=" * 65)
    print(" OVERALL PERFORMANCE")
    print("=" * 65)

    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-score:            {f1:.4f}")
    print(f"False positive rate: {false_positive_rate:.4f}")

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
    # CATEGORY ANALYSIS
    # ========================================================

    evaluation_df = df.copy()

    evaluation_df["predicted_label"] = y_pred

    evaluation_df["malicious_probability"] = probabilities

    print("\n" + "=" * 65)
    print(" CATEGORY PERFORMANCE")
    print("=" * 65)

    category_results = []

    for category, group in evaluation_df.groupby("category"):

        category_accuracy = accuracy_score(
            group["label"],
            group["predicted_label"]
        )

        false_positives = int(
            (
                (group["label"] == 0)
                & (group["predicted_label"] == 1)
            ).sum()
        )

        false_negatives = int(
            (
                (group["label"] == 1)
                & (group["predicted_label"] == 0)
            ).sum()
        )

        category_results.append({
            "category": category,
            "samples": len(group),
            "accuracy": category_accuracy,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        })

        print(
            f"{category}: "
            f"Accuracy {category_accuracy:.2%} | "
            f"FP: {false_positives} | "
            f"FN: {false_negatives}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "model": "RoBERTa",
        "dataset": DATASET_PATH.name,
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

    results_path = (
        RESULTS_DIR / "roberta_contextual_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    category_path = (
        RESULTS_DIR / "roberta_contextual_categories.csv"
    )

    pd.DataFrame(category_results).to_csv(
        category_path,
        index=False
    )

    predictions_path = (
        RESULTS_DIR / "roberta_contextual_predictions.csv"
    )

    evaluation_df.to_csv(
        predictions_path,
        index=False
    )

    print("\n" + "=" * 65)
    print(" EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Overall results: {results_path}")
    print(f"Category results: {category_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()