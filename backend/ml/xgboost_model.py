# ==========================================
# SENTINELAI - XGBOOST CLASSIFICATION
# ==========================================

import os
import time
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

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

os.makedirs(MODEL_DIR, exist_ok=True)

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

print("Training Shape:", X_train.shape)

print("Validation Shape:", X_validation.shape)

# ==========================================
# 3. CALCULATE CLASS WEIGHT
# ==========================================

negative = np.sum(y_train == 0)

positive = np.sum(y_train == 1)

scale_weight = negative / positive

print("\nClass Weight:", scale_weight)

# ==========================================
# 4. INITIALIZE XGBOOST
# ==========================================

model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_weight,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    random_state=42,

    n_jobs=4

)

# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining XGBoost...")

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print(
    "\nTraining completed in:",
    round(training_time, 2),
    "seconds"
)

# ==========================================
# 6. VALIDATION PREDICTION
# ==========================================

predictions = model.predict(
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

print("\n=================================")
print("XGBOOST PERFORMANCE")
print("=================================")

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

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        predictions
    )
)

# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    )
)

print("\nXGBoost model saved successfully!")

# ==========================================
# 9. SAVE RESULTS
# ==========================================

results = pd.DataFrame([{

    "Model": "XGBoost",

    "Accuracy": accuracy,

    "Precision": precision,

    "Recall": recall,

    "F1 Score": f1,

    "Training Time": training_time

}])

results.to_csv(
    os.path.join(
        MODEL_DIR,
        "xgboost_results.csv"
    ),
    index=False
)

print("\nXGBoost experiment completed!")
