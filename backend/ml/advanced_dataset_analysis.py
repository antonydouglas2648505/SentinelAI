import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SENTINELAI - ADVANCED DATASET AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "backend" / "data" / "raw"

FILES = {
    "Training": DATA_DIR / "train.csv",
    "Validation": DATA_DIR / "validation.csv",
    "Testing": DATA_DIR / "test.csv",
}

LABEL_NAMES = {
    0: "Benign",
    1: "Malicious"
}


def section(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def normalize_text(text):
    """Normalize text for duplicate detection."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def load_datasets():
    datasets = {}

    section("LOADING DATASETS")

    for name, path in FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}"
            )

        df = pd.read_csv(path)

        required_columns = {"text", "label"}

        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"{name} dataset must contain "
                f"'text' and 'label' columns."
            )

        datasets[name] = df.copy()

        print(
            f"{name}: {df.shape[0]:,} records | "
            f"{df.shape[1]} columns"
        )

    return datasets


def analyze_dataset_structure(datasets):

    section("1. DATASET STRUCTURE")

    for name, df in datasets.items():

        print(f"\n{name.upper()}")

        print(f"Total records: {len(df):,}")
        print(f"Columns: {list(df.columns)}")

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nDuplicate complete rows:")
        print(df.duplicated().sum())

        print("\nDuplicate text records:")
        print(df["text"].duplicated().sum())


def analyze_class_distribution(datasets):

    section("2. CLASS DISTRIBUTION")

    for name, df in datasets.items():

        print(f"\n{name.upper()}")

        counts = df["label"].value_counts().sort_index()

        for label, count in counts.items():

            percentage = count / len(df) * 100

            label_name = LABEL_NAMES.get(
                label,
                f"Unknown ({label})"
            )

            print(
                f"{label_name}: {count:,} "
                f"({percentage:.2f}%)"
            )


def analyze_text_lengths(datasets):

    section("3. TEXT LENGTH ANALYSIS")

    for name, df in datasets.items():

        lengths = df["text"].fillna("").astype(str).str.split().str.len()

        print(f"\n{name.upper()}")

        print(f"Minimum words: {lengths.min()}")
        print(f"Maximum words: {lengths.max()}")
        print(f"Average words: {lengths.mean():.2f}")
        print(f"Median words: {lengths.median():.2f}")

        print("\nPercentiles:")

        for percentile in [50, 75, 90, 95, 99]:

            value = np.percentile(lengths, percentile)

            print(f"{percentile}th percentile: {value:.0f}")


def analyze_normalized_duplicates(datasets):

    section("4. NORMALIZED DUPLICATE ANALYSIS")

    normalized = {}

    for name, df in datasets.items():

        normalized[name] = df["text"].fillna("").apply(
            normalize_text
        )

        duplicate_count = normalized[name].duplicated().sum()

        print(
            f"{name}: {duplicate_count:,} "
            f"normalized duplicate records"
        )


def analyze_cross_dataset_leakage(datasets):

    section("5. CROSS-DATASET LEAKAGE ANALYSIS")

    normalized_sets = {}

    for name, df in datasets.items():

        normalized_sets[name] = set(
            df["text"].dropna().apply(normalize_text)
        )

    comparisons = [
        ("Training", "Validation"),
        ("Training", "Testing"),
        ("Validation", "Testing"),
    ]

    for first, second in comparisons:

        overlap = (
            normalized_sets[first]
            & normalized_sets[second]
        )

        print(f"\n{first} vs {second}")

        print(f"Overlapping records: {len(overlap):,}")

        if overlap:
            print("WARNING: Potential data leakage detected!")

            for example in list(overlap)[:3]:
                print(f"Example: {example[:150]}")


def analyze_conflicting_labels(datasets):

    section("6. CONFLICTING LABEL ANALYSIS")

    for name, df in datasets.items():

        cleaned = df.dropna(subset=["text", "label"]).copy()

        cleaned["normalized_text"] = cleaned["text"].apply(
            normalize_text
        )

        conflicts = (
            cleaned.groupby("normalized_text")["label"]
            .nunique()
        )

        conflicting_records = conflicts[conflicts > 1]

        print(f"\n{name}")

        print(
            f"Texts with conflicting labels: "
            f"{len(conflicting_records):,}"
        )


def analyze_sample_examples(datasets):

    section("7. SAMPLE DATA INSPECTION")

    for name, df in datasets.items():

        print(f"\n{name.upper()}")

        for label in [0, 1]:

            subset = df[df["label"] == label]

            if subset.empty:
                continue

            label_name = LABEL_NAMES.get(label, str(label))

            print(f"\n{label_name} examples:")

            for index, row in subset.head(3).iterrows():

                text = str(row["text"]).replace("\n", " ")

                print(f"\nSample {index}:")
                print(text[:300])


def generate_audit_summary(datasets):

    section("8. FINAL AUDIT SUMMARY")

    total_records = sum(
        len(df) for df in datasets.values()
    )

    print(f"Total records: {total_records:,}")

    print("\nDataset sizes:")

    for name, df in datasets.items():

        print(f"{name}: {len(df):,}")

    combined = pd.concat(
        datasets.values(),
        ignore_index=True
    )

    print("\nCombined class distribution:")

    print(
        combined["label"]
        .value_counts()
        .sort_index()
    )

    print("\nDataset audit completed successfully.")


def main():

    print("\n")
    print("=" * 65)
    print(" SENTINELAI - ADVANCED DATASET AUDIT")
    print("=" * 65)

    datasets = load_datasets()

    analyze_dataset_structure(datasets)

    analyze_class_distribution(datasets)

    analyze_text_lengths(datasets)

    analyze_normalized_duplicates(datasets)

    analyze_cross_dataset_leakage(datasets)

    analyze_conflicting_labels(datasets)

    analyze_sample_examples(datasets)

    generate_audit_summary(datasets)


if __name__ == "__main__":
    main()