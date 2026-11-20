import os
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

ML_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(ML_DIR)

DATA_DIR = os.path.join(
    BACKEND_DIR, "data", "processed"
)

X = load_npz(
    os.path.join(DATA_DIR, "X_train.npz")
)

y = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC(
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=4
    )

}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    print("\nEvaluating:", name)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ],
        n_jobs=1
    )

    results.append({

        "Model": name,

        "Accuracy": scores["test_accuracy"].mean(),

        "Precision": scores["test_precision"].mean(),

        "Recall": scores["test_recall"].mean(),

        "F1 Score": scores["test_f1"].mean()

    })

    print("F1:", scores["test_f1"].mean())

results_df = pd.DataFrame(results)

print("\nCROSS VALIDATION RESULTS")

print(results_df)

results_df.to_csv(
    os.path.join(
        BACKEND_DIR,
        "data",
        "processed",
        "cross_validation_results.csv"
    ),
    index=False
)