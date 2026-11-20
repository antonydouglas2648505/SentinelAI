# ==========================================
# SENTINELAI - SHAP EXPLAINABILITY
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy.sparse import load_npz, hstack

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
    "model_comparison",
    "shap_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD MODEL
# ==========================================

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "logistic_regression.pkl"
    )
)

print("\nLogistic Regression loaded!")

# ==========================================
# 3. LOAD FEATURE MATRICES
# ==========================================

X_train = load_npz(
    os.path.join(DATA_DIR, "X_train.npz")
)

X_validation = load_npz(
    os.path.join(DATA_DIR, "X_validation.npz")
)

# Use a small representative background sample.
# Convert only these selected rows to dense arrays.

rng = np.random.default_rng(42)

background_indices = rng.choice(
    X_train.shape[0],
    size=min(40, X_train.shape[0]),
    replace=False
)

sample_indices = rng.choice(
    X_validation.shape[0],
    size=min(20, X_validation.shape[0]),
    replace=False
)

background = X_train[
    background_indices
].toarray()

samples = X_validation[
    sample_indices
].toarray()

# ==========================================
# 4. LOAD VECTORIZERS
# ==========================================

word_vectorizer = joblib.load(
    os.path.join(
        DATA_DIR,
        "word_vectorizer.pkl"
    )
)

char_vectorizer = joblib.load(
    os.path.join(
        DATA_DIR,
        "char_vectorizer.pkl"
    )
)

feature_names = np.concatenate([

    word_vectorizer.get_feature_names_out(),

    np.array([
        "char_" + x
        for x in char_vectorizer.get_feature_names_out()
    ])

])

# ==========================================
# 5. CREATE SHAP EXPLAINER
# ==========================================

print("\nCreating SHAP explainer...")

explainer = shap.LinearExplainer(
    model,
    background,
    feature_perturbation="interventional"
)

# ==========================================
# 6. CALCULATE SHAP VALUES
# ==========================================

print("\nCalculating SHAP values...")

shap_values = explainer(
    samples,
    check_additivity=False
)

# ==========================================
# 7. GLOBAL FEATURE IMPORTANCE
# ==========================================

importance = np.abs(
    shap_values.values
).mean(axis=0)

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 20 IMPORTANT FEATURES")

print(
    importance_df.head(20).to_string(index=False)
)

importance_df.head(30).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_features.csv"
    ),
    index=False
)

# ==========================================
# 8. SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP Summary Plot...")

shap.plots.beeswarm(
    shap_values,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "shap_summary.png"
    ),
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 9. SHAP BAR PLOT
# ==========================================

shap.plots.bar(
    shap_values,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "shap_feature_importance.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("\n================================")
print(" SHAP ANALYSIS COMPLETED")
print("================================")