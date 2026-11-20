# ==========================================
# SENTINELAI - ENSEMBLE LEARNING
# ==========================================

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================

ML_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(ML_DIR)

DATA_DIR = os.path.join(
    BACKEND_DIR, "data", "processed"
)

MODEL_DIR = os.path.join(
    BACKEND_DIR, "models"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(BACKEND_DIR),
    "research",
    "model_comparison"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading datasets...")

X_train = load_npz(
    os.path.join(DATA_DIR, "X_train.npz")
)

X_validation = load_npz(
    os.path.join(DATA_DIR, "X_validation.npz")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_validation = np.load(
    os.path.join(DATA_DIR, "y_validation.npy")
)

print("Training:", X_train.shape)
print("Validation:", X_validation.shape)


# ==========================================
# 3. DEFINE BASE MODELS
# ==========================================

logistic = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    random_state=42
)

naive_bayes = MultinomialNB()

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=4
)

xgboost = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)


# ==========================================
# 4. CREATE ENSEMBLE MODEL
# ==========================================

ensemble = VotingClassifier(

    estimators=[
        ("lr", logistic),
        ("nb", naive_bayes),
        ("rf", random_forest),
        ("xgb", xgboost)
    ],

    voting="soft",

    weights=[2, 1, 1, 2],

    n_jobs=1
)


# ==========================================
# 5. TRAIN ENSEMBLE
# ==========================================

print("\nTraining Ensemble Model...")

start = time.time()

ensemble.fit(
    X_train,
    y_train
)

training_time = time.time() - start

print(
    "\nTraining completed:",
    round(training_time, 2),
    "seconds"
)


# ==========================================
# 6. VALIDATION PREDICTION
# ==========================================

predictions = ensemble.predict(
    X_validation
)

# ==========================================
# 7. EVALUATION
# ==========================================

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

print("\n========== ENSEMBLE RESULTS ==========")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        predictions,
        target_names=["Benign", "Malicious"],
        zero_division=0
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_validation,
    predictions,
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

plt.title("SentinelAI Ensemble Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_confusion_matrix.png"
    )
)

plt.close()


# ==========================================
# 9. SAVE ENSEMBLE MODEL
# ==========================================

joblib.dump(
    ensemble,
    os.path.join(
        MODEL_DIR,
        "ensemble_model.pkl"
    )
)

print("\nEnsemble model saved successfully!")


# ==========================================
# 10. SAVE RESULTS
# ==========================================

results = pd.DataFrame([{

    "Model": "Soft Voting Ensemble",

    "Accuracy": accuracy,

    "Precision": precision,

    "Recall": recall,

    "F1 Score": f1,

    "Training Time": training_time

}])

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_results.csv"
    ),
    index=False
)

print("\nEnsemble experiment completed!")