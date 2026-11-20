import os
import time
import joblib
import numpy as np

from scipy.sparse import load_npz

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Directories
ML_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(ML_DIR)

DATA_DIR = os.path.join(
    BACKEND_DIR, "data", "processed"
)

MODEL_DIR = os.path.join(
    BACKEND_DIR, "models"
)

# Load data
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

# XGBoost
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)

# Parameter search space
parameters = {

    "n_estimators": [100, 200, 300],

    "max_depth": [3, 5, 7],

    "learning_rate": [0.01, 0.05, 0.1],

    "subsample": [0.7, 0.8, 1.0],

    "colsample_bytree": [0.7, 0.8, 1.0],

    "min_child_weight": [1, 3, 5]

}

# Randomized Search
search = RandomizedSearchCV(

    estimator=model,

    param_distributions=parameters,

    n_iter=12,

    scoring="f1",

    cv=3,

    verbose=2,

    random_state=42,

    n_jobs=1

)

print("\nStarting Hyperparameter Optimization...")

start = time.time()

search.fit(
    X_train,
    y_train
)

print(
    "\nOptimization Time:",
    round(time.time() - start, 2),
    "seconds"
)

# Best parameters
print("\nBest Parameters:")

print(search.best_params_)

print("\nBest Cross Validation F1:")

print(search.best_score_)

# Evaluate using validation set
best_model = search.best_estimator_

predictions = best_model.predict(
    X_validation
)

print("\nValidation Results")

print(
    "Accuracy:",
    accuracy_score(y_validation, predictions)
)

print(
    "Precision:",
    precision_score(
        y_validation,
        predictions,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_validation,
        predictions,
        zero_division=0
    )
)

print(
    "F1 Score:",
    f1_score(
        y_validation,
        predictions,
        zero_division=0
    )
)

# Save optimized model
joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "optimized_xgboost.pkl"
    )
)

print("\nOptimized model saved successfully!")