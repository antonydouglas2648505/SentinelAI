import json
import torch
import numpy as np
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
    confusion_matrix,
    classification_report
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROJECT_DIR = BASE_DIR.parent

MODELS = {
    "Original RoBERTa": BASE_DIR / "models" / "roberta",
    "Augmented RoBERTa": BASE_DIR / "models" / "roberta_augmented"
}

EVALUATION_DATASETS = {
    "Prompt Injection": (
        BASE_DIR / "data" / "evaluation"
        / "prompt_injection_test.csv"
    ),

    "Contextual": (
        BASE_DIR / "data" / "augmentation"
        / "contextual_samples.csv"
    )
}

RESULT_DIR = (
    PROJECT_DIR / "research" / "augmentation_results"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5
BATCH_SIZE = 8
MAX_LENGTH = 256


# ==========================================
# MODEL LOADING
# ==========================================

def load_model(model_path):

    print(f"\nLoading: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return tokenizer, model


# ==========================================
# PREDICTION
# ==========================================

def predict_dataset(tokenizer, model, texts):

    probabilities = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
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

            batch_probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[:, 1]

        probabilities.extend(
            batch_probabilities.cpu().numpy().tolist()
        )

    probabilities = np.array(probabilities)

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return probabilities, predictions


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "false_positive_rate": float(
            fp / (fp + tn)
        ) if (fp + tn) > 0 else 0.0,

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        "confusion_matrix": cm.tolist()
    }


# ==========================================
# MAIN EVALUATION
# ==========================================

def main():

    print("\n" + "=" * 65)
    print(" SENTINELAI - ROBERTA MODEL COMPARISON")
    print("=" * 65)

    print("\nDevice:", DEVICE)

    all_results = {}
    all_predictions = []

    loaded_models = {}

    # Load both models.
    for model_name, model_path in MODELS.items():

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        loaded_models[model_name] = load_model(
            model_path
        )

    # Evaluate each dataset.
    for dataset_name, dataset_path in EVALUATION_DATASETS.items():

        print("\n" + "=" * 65)
        print(f" DATASET: {dataset_name}")
        print("=" * 65)

        if not dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        df = pd.read_csv(dataset_path)

        required = {"text", "label"}

        if not required.issubset(df.columns):

            raise ValueError(
                f"{dataset_name} must contain text and label columns."
            )

        if df["text"].isna().any():

            raise ValueError(
                f"{dataset_name} contains missing text."
            )

        df["label"] = pd.to_numeric(
            df["label"],
            errors="raise"
        ).astype(int)

        if not set(df["label"].unique()).issubset({0, 1}):

            raise ValueError(
                f"{dataset_name} contains invalid labels."
            )

        texts = df["text"].astype(str).tolist()

        y_true = df["label"].to_numpy()

        print("\nTotal samples:", len(df))

        all_results[dataset_name] = {}

        for model_name, (tokenizer, model) in loaded_models.items():

            print(f"\nEvaluating {model_name}...")

            probabilities, predictions = predict_dataset(
                tokenizer,
                model,
                texts
            )

            metrics = calculate_metrics(
                y_true,
                predictions
            )

            all_results[dataset_name][model_name] = metrics

            print("\nMETRICS")

            for metric, value in metrics.items():

                if metric != "confusion_matrix":

                    print(f"{metric}: {value}")

            print("\nConfusion Matrix:")

            print(
                np.array(
                    metrics["confusion_matrix"]
                )
            )

            print("\nClassification Report:")

            print(
                classification_report(
                    y_true,
                    predictions,
                    labels=[0, 1],
                    target_names=["Benign", "Malicious"],
                    zero_division=0
                )
            )

            # Store individual predictions.
            for index in range(len(df)):

                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "text": texts[index],
                    "actual_label": int(y_true[index]),
                    "predicted_label": int(predictions[index]),
                    "malicious_probability": float(
                        probabilities[index]
                    ),
                    "correct": bool(
                        y_true[index] == predictions[index]
                    )
                }

                if "category" in df.columns:

                    record["category"] = df.iloc[index]["category"]

                all_predictions.append(record)

    # ======================================
    # SAVE RESULTS
    # ======================================

    results_file = (
        RESULT_DIR / "roberta_model_comparison.json"
    )

    predictions_file = (
        RESULT_DIR / "roberta_model_comparison_predictions.csv"
    )

    with open(results_file, "w") as file:

        json.dump(
            all_results,
            file,
            indent=4
        )

    pd.DataFrame(all_predictions).to_csv(
        predictions_file,
        index=False
    )

    # ======================================
    # COMPARISON SUMMARY
    # ======================================

    print("\n" + "=" * 65)
    print(" FINAL MODEL COMPARISON")
    print("=" * 65)

    summary = []

    for dataset_name, models in all_results.items():

        for model_name, metrics in models.items():

            summary.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1_score"],
                "False Positive Rate": metrics[
                    "false_positive_rate"
                ]
            })

    summary_df = pd.DataFrame(summary)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    summary_file = (
        RESULT_DIR / "roberta_model_comparison_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 65)
    print(" COMPARISON COMPLETED")
    print("=" * 65)

    print("\nResults:")
    print(results_file)

    print("\nSummary:")
    print(summary_file)

    print("\nPredictions:")
    print(predictions_file)


if __name__ == "__main__":
    main()