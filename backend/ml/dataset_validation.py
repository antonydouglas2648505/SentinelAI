import pandas as pd

# Load datasets
train = pd.read_csv("data/raw/train.csv")
validation = pd.read_csv("data/raw/validation.csv")
test = pd.read_csv("data/raw/test.csv")

datasets = {
    "Training": train,
    "Validation": validation,
    "Testing": test
}

print("\n========== SENTINELAI DATA VALIDATION ==========")

# 1. Class distribution
for name, df in datasets.items():

    print(f"\n{name} Dataset")

    print("Total Samples:", len(df))

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    print("\nClass Percentage:")
    print(df["label"].value_counts(normalize=True) * 100)


# 2. Duplicate analysis
print("\n========== DUPLICATE ANALYSIS ==========")

for name, df in datasets.items():

    print(
        name,
        "Duplicates:",
        df["text"].duplicated().sum()
    )


# 3. Cross-dataset leakage
print("\n========== CROSS-SPLIT DUPLICATES ==========")

train_texts = set(train["text"].astype(str))
val_texts = set(validation["text"].astype(str))
test_texts = set(test["text"].astype(str))

print("Train-Validation:",
      len(train_texts & val_texts))

print("Train-Test:",
      len(train_texts & test_texts))

print("Validation-Test:",
      len(val_texts & test_texts))


# 4. Text length
print("\n========== TEXT LENGTH ANALYSIS ==========")

for name, df in datasets.items():

    lengths = df["text"].astype(str).str.len()

    print(f"\n{name}")

    print("Average:", lengths.mean())
    print("Minimum:", lengths.min())
    print("Maximum:", lengths.max())

print("\n========== VALIDATION COMPLETED ==========")
