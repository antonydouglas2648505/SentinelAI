import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PREDICTIONS_FILE = (
    ROOT_DIR
    / "research"
    / "evaluation_results"
    / "roberta_contextual_predictions.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "research"
    / "evaluation_results"
    / "contextual_false_positive_review.csv"
)


def main():

    print("\n" + "=" * 65)
    print(" SENTINELAI - CONTEXTUAL FALSE POSITIVE ANALYSIS")
    print("=" * 65)

    if not PREDICTIONS_FILE.exists():
        print("\nERROR: Predictions file not found.")
        return

    df = pd.read_csv(PREDICTIONS_FILE)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    # Identify false positives.
    false_positives = df[
        (df["label"] == 0) &
        (df["predicted_label"] == 1)
    ].copy()

    print(f"\nTotal false positives: {len(false_positives)}")

    if false_positives.empty:
        print("\nNo false positives found!")
        return

    print("\nFALSE POSITIVE DETAILS")
    print("-" * 65)

    for index, row in false_positives.iterrows():

        print(f"\nSample: {index + 1}")
        print(f"Category: {row['category']}")
        print(f"Actual label: {row['label']}")
        print(f"Predicted label: {row['predicted_label']}")

        probability_column = (
            "malicious_probability"
            if "malicious_probability" in df.columns
            else None
        )

        if probability_column:
            print(
                f"Malicious probability: "
                f"{row[probability_column]}"
            )

        print(f"Text: {row['text']}")

    false_positives.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print(" ANALYSIS COMPLETED")
    print("=" * 65)

    print(f"\nReview file saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()