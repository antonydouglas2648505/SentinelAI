# ==================================================
# SENTINELAI - MACHINE LEARNING MODEL DEVELOPMENT
# ==================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. DIRECTORY CONFIGURATION
# ==================================================

ML_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(ML_DIR)

DATA_DIR = os.path.join(
    BACKEND_DIR, "data", "processed"
)

MODEL_DIR = os.path.join(
    BACKEND_DIR, "models"
)

RESEARCH_DIR = os.path.join(
    os.path.dirname(BACKEND_DIR),
    "research",
    "model_comparison"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESEARCH_DIR, exist_ok=True)

print("\n========================================")
print(" SENTINELAI - ML MODEL TRAINING")
print("========================================")

# ==================================================
# 2. LOAD PREPROCESSED FEATURES
# ==================================================

print("\nLoading preprocessed datasets...")

X_train = load_npz(
    os.path.join(DATA_DIR, "X_train.npz")
)

X_validation = load_npz(
    os.path.join(DATA_DIR, "X_validation.npz")
)

X_test = load_npz(
    os.path.join(DATA_DIR, "X_test.npz")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_validation = np.load(
    os.path.join(DATA_DIR, "y_validation.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

print("\nDataset Information")

print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)

# ==================================================
# 3. DEFINE MACHINE LEARNING MODELS
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=100,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    )
}

# ==================================================
# 4. TRAIN AND EVALUATE MODELS
# ==================================================

results = []

trained_models = {}

best_model_name = None
best_model = None
best_f1 = -1

for name, model in models.items():

    print("\n========================================")
    print("TRAINING:", name)
    print("========================================")

    # Training
    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    # Validation prediction
    prediction_start = time.time()

    predictions = model.predict(
        X_validation
    )

    prediction_time = time.time() - prediction_start

    # Evaluation metrics
    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    fnr = fn / (fn + tp) if (fn + tp) else 0

    # Store results
    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "False Positive Rate": fpr,

        "False Negative Rate": fnr,

        "Training Time": training_time,

        "Prediction Time": prediction_time

    })

    trained_models[name] = model

    print("Accuracy:", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall:", round(recall, 4))

    print("F1 Score:", round(f1, 4))

    print("False Positive Rate:", round(fpr, 4))

    print("False Negative Rate:", round(fnr, 4))

    print("Training Time:", round(training_time, 3), "seconds")

    print("\nClassification Report")

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=["Benign", "Malicious"],
            zero_division=0
        )
    )

    # Select best model using validation F1
    if f1 > best_f1:

        best_f1 = f1

        best_model_name = name

        best_model = model

# ==================================================
# 5. MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========================================")
print("MODEL COMPARISON RESULTS")
print("========================================")

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    os.path.join(
        RESEARCH_DIR,
        "model_comparison.csv"
    ),
    index=False
)

# ==================================================
# 6. SAVE ALL TRAINED MODELS
# ==================================================

for name, model in trained_models.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, filename)
    )

# ==================================================
# 7. SELECT BEST MODEL
# ==================================================

print("\n========================================")
print("BEST MODEL SELECTION")
print("========================================")

print("Best Model:", best_model_name)

print("Validation F1 Score:", round(best_f1, 4))

joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )
)

with open(
    os.path.join(
        MODEL_DIR,
        "best_model_name.txt"
    ),
    "w"
) as file:

    file.write(best_model_name)

# ==================================================
# 8. FINAL TEST EVALUATION
# ==================================================

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================")

test_predictions = best_model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

print("Selected Model:", best_model_name)

print("Test Accuracy:", round(test_accuracy, 4))

print("Test Precision:", round(test_precision, 4))

print("Test Recall:", round(test_recall, 4))

print("Test F1 Score:", round(test_f1, 4))

print("\nFinal Classification Report")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Benign", "Malicious"],
        zero_division=0
    )
)

# ==================================================
# 9. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malicious"],
    yticklabels=["Benign", "Malicious"]
)

plt.title(
    f"SentinelAI - {best_model_name} Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESEARCH_DIR,
        "best_model_confusion_matrix.png"
    )
)

plt.close()

# ==================================================
# 10. MODEL COMPARISON GRAPH
# ==================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

comparison = results_df.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("SentinelAI - ML Model Performance Comparison")

plt.xticks(rotation=20)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESEARCH_DIR,
        "model_performance_comparison.png"
    )
)

plt.close()

# ==================================================
# 11. SAVE FINAL TEST RESULTS
# ==================================================

test_results = pd.DataFrame([{

    "Selected Model": best_model_name,

    "Accuracy": test_accuracy,

    "Precision": test_precision,

    "Recall": test_recall,

    "F1 Score": test_f1

}])

test_results.to_csv(
    os.path.join(
        RESEARCH_DIR,
        "final_test_results.csv"
    ),
    index=False
)

print("\n========================================")
print(" SENTINELAI ML TRAINING COMPLETED")
print("========================================")
