"""
training_models.py

Load preprocessed TTM survey data, train multiple classification models
to predict relapse risk, evaluate them on a held-out test set, and save
the best-performing pipeline to disk for later use.

Models trained:
 - Logistic Regression
 - Random Forest
 - Gradient Boosting
 - Multi-layer Perceptron

Outputs:
 - models/best_model.pkl       : serialized best pipeline
 - models/test_features.csv    : held-out test features
 - models/test_target.csv      : held-out test labels

Usage:
    python training_models.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- Paths & constants ---
PREPROCESSED_CSV = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\preprocessed_ttm_data.csv"
MODEL_DIR       = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models"
TEST_FEAT_FILE  = os.path.join(MODEL_DIR, "test_features.csv")
TEST_TARG_FILE  = os.path.join(MODEL_DIR, "test_target.csv")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    """Load preprocessed data from CSV."""
    return pd.read_csv(path)

def prepare_features(df, target_col="relapse_risk_tag"):
    """
    Split off target, encode it, and select numeric features.
    Returns X (DataFrame), y (Series).
    """
    y = df[target_col].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # save label mapping alongside model
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    # select numeric columns only
    X = df.select_dtypes(include=[np.number]).drop(columns=[])  # drop none
    return X, y_enc

def train_and_select(X_train, y_train, X_test, y_test):
    """
    Train several pipelines, evaluate on test set,
    pick best by accuracy, and return (best_name, best_pipeline).
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest":      RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoost":     GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "MLPClassifier":     MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, random_state=RANDOM_STATE)
    }

    best_acc = -1.0
    best_name, best_pipe = None, None

    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",     clf)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:15s} test accuracy: {acc:.3f}")
        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    print(f"\nBest model: {best_name} (accuracy={best_acc:.3f})")
    return best_name, best_pipe

def main():
    # 1. Load
    df = load_data(PREPROCESSED_CSV)

    # 2. Prepare
    X, y = prepare_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 4. Persist test set for external evaluation
    X_test.to_csv(TEST_FEAT_FILE, index=False)
    pd.Series(y_test, name="target").to_csv(TEST_TARG_FILE, index=False)

    # 5. Train & select
    _, best_pipe = train_and_select(X_train, y_train, X_test, y_test)

    # 6. Save best pipeline
    joblib.dump(best_pipe, BEST_MODEL_FILE)
    print(f"✅ Saved best model to {BEST_MODEL_FILE}")

if __name__ == "__main__":
    main()
