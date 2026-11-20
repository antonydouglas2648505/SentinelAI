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

VALIDATION_PATH = BASE_DIR / "data" / "raw" / "validation.csv"

MODELS = {
    "Original RoBERTa": BASE_DIR / "models" / "roberta",
    "Augmented RoBERTa": BASE_DIR / "models" / "roberta_augmented"
}

RESULT_DIR = BASE_DIR.parent / "research" / "augmentation_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
MAX_LENGTH = 256

THRESHOLDS = [
    0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70
]


def get_probabilities(model_path, texts):

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    model.to(DEVICE)
    model.eval()

    probabilities_list = []

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

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probabilities_list


def calculate_metrics(labels, probabilities, threshold):

    predictions = [
        1 if probability >= threshold else 0
        for probability in probabilities
    ]

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
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
        "false_negatives": int(fn),
        "false_positives": int(fp)
    }


def main():

    print("=" * 65)
    print(" SENTINELAI - CLASSIFICATION THRESHOLD ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(VALIDATION_PATH)

    df = df.dropna(subset=["text", "label"])

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    print(f"\nValidation samples: {len(df)}")
    print(f"Device: {DEVICE}")

    all_results = []

    for model_name, model_path in MODELS.items():

        print(f"\nEvaluating {model_name}...")

        probabilities = get_probabilities(
            model_path,
            texts
        )

        for threshold in THRESHOLDS:

            metrics = calculate_metrics(
                labels,
                probabilities,
                threshold
            )

            metrics["model"] = model_name

            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    results_df = results_df[
        [
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
            "false_positives",
            "false_negatives"
        ]
    ]

    output_path = RESULT_DIR / "threshold_analysis.csv"

    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print(" THRESHOLD COMPARISON")
    print("=" * 65)

    for model_name in MODELS:

        print(f"\n{model_name}")
        print("-" * 65)

        model_results = results_df[
            results_df["model"] == model_name
        ]

        print(
            model_results.drop(columns=["model"]).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()