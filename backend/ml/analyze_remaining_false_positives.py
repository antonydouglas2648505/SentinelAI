import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    BASE_DIR.parent
    / "research"
    / "augmentation_results"
    / "roberta_model_comparison_predictions.csv"
)


def main():

    print("\n" + "=" * 65)
    print(" SENTINELAI - REMAINING FALSE POSITIVE ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(PREDICTIONS_FILE)

    augmented = df[
        df["model"] == "Augmented RoBERTa"
    ]

    false_positives = augmented[
        (augmented["actual_label"] == 0) &
        (augmented["predicted_label"] == 1)
    ]

    print(
        "\nRemaining false positives:",
        len(false_positives)
    )

    if false_positives.empty:

        print("\nNo false positives found!")
        return

    for index, row in false_positives.iterrows():

        print("\n" + "-" * 65)

        print("Dataset:", row["dataset"])

        print("Category:", row.get("category", "Unknown"))

        print("Actual label:", row["actual_label"])

        print("Predicted label:", row["predicted_label"])

        print(
            "Malicious probability:",
            f"{row['malicious_probability']:.4f}"
        )

        print("\nText:")

        print(row["text"])

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()