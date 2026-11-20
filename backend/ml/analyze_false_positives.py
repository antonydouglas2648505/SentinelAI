import pandas as pd
from pathlib import Path


# ============================================================
# SENTINELAI - FALSE POSITIVE ANALYZER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "research"
    / "evaluation_results"
    / "roberta_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "research"
    / "evaluation_results"
    / "false_positive_review.csv"
)


def main():

    print("\n" + "=" * 60)
    print(" SENTINELAI - FALSE POSITIVE ANALYSIS")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # False positives:
    # Actual label = 0
    # Predicted label = 1

    false_positives = df[
        (df["label"] == 0)
        & (df["predicted_label"] == 1)
    ].copy()

    print(f"\nTotal evaluated samples: {len(df)}")

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    print(
        f"Benign samples: "
        f"{(df['label'] == 0).sum()}"
    )

    if false_positives.empty:

        print("\nNo false positives found!")

        return

    print("\n" + "-" * 60)
    print(" INCORRECTLY FLAGGED BENIGN INPUTS")
    print("-" * 60)

    for index, (_, row) in enumerate(
        false_positives.iterrows(),
        start=1
    ):

        print(f"\nFALSE POSITIVE #{index}")

        print(f"Category: {row['category']}")

        print(
            f"Malicious probability: "
            f"{row['malicious_probability']:.2%}"
        )

        print(f"Text:\n{row['text']}")

    # Save false positives for manual review.

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    false_positives.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print(" ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"\nReview file saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()