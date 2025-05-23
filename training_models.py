#!/usr/bin/env python3
"""
training_models.py

Load your two preprocessed tables from SQLite, merge them into one DataFrame,
train three classifiers (LogisticRegression, RandomForest, GradientBoosting),
evaluate on a held-out split, and save out:

  - models/best_model.pkl
  - models/label_encoder.pkl
  - models/test_features.csv
  - models/test_target.csv

Usage:
    python training_models.py
"""
import os
import sqlite3
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import accuracy_score

# ── PATHS & CONSTANTS ────────────────────────────────────────────────────────────
DB_PATH        = "database/ttm_database.db"
MODEL_DIR      = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_F   = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENC_F    = os.path.join(MODEL_DIR, "label_encoder.pkl")
TEST_FEAT_CSV  = os.path.join(MODEL_DIR, "test_features.csv")
TEST_TARG_CSV  = os.path.join(MODEL_DIR, "test_target.csv")

TEST_SIZE      = 0.2
RANDOM_STATE   = 42


def load_and_merge() -> pd.DataFrame:
    """
    Load 'demographics' and 'hair_pulling_behaviours_patterns' tables,
    merge them on 'id', and return a single DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    demo = (
        pd.read_sql("SELECT * FROM demographics", conn)
          .set_index("id")
    )
    beh = (
        pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns", conn)
          .set_index("id")
    )
    conn.close()

    # Inner‐join: keep only users present in both tables
    df = demo.join(beh, how="inner", lsuffix="_demo", rsuffix="_beh", on="id", validate="one_to_one")
    # Drop rows missing our target
    return df.dropna(subset=["relapse_risk_tag"])


def prepare_dataset(df: pd.DataFrame):
    """
    Separate features X from target y, encode y, persist the label encoder.

    Returns
    -------
    X : pd.DataFrame of numeric features
    y : np.ndarray of encoded labels
    """
    # 1) Encode target
    y = df["relapse_risk_tag"].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LABEL_ENC_F)

    # 2) Select only numeric columns for X
    X = df.select_dtypes(include=[np.number]).fillna(0)

    return X, y_enc


def train_and_select(X_train, y_train, X_test, y_test):
    """
    Train three pipelines (scaling + classifier), evaluate on the held-out set,
    and return the best-fitting pipeline.
    """
    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest"      : RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            min_samples_leaf=1,
            max_features="sqrt"
        ),
        "GradientBoosting"  : GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    best_name, best_acc, best_pipe = None, -np.inf, None

    for name, clf in classifiers.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:18s} test accuracy: {acc:.3f}")
        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    print(f"\n✅ Best model: {best_name} (accuracy={best_acc:.3f})\n")
    return best_pipe


def main():
    """Load data, prepare, train+, select, and persist the best model."""
    print("🔍 Loading & merging tables…")
    df = load_and_merge()

    print("⚙️  Preparing features and target…")
    X, y = prepare_dataset(df)

    print(f"🔀 Splitting into train/test (test_size={TEST_SIZE})…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Persist the held‐out split
    X_test.to_csv(TEST_FEAT_CSV, index=False)
    pd.Series(y_test, name="target").to_csv(TEST_TARG_CSV, index=False)

    print("🏋️  Training and selecting best pipeline…")
    best_pipe = train_and_select(X_train, y_train, X_test, y_test)

    print(f"💾 Saving best pipeline to {BEST_MODEL_F}")
    joblib.dump(best_pipe, BEST_MODEL_F)

    print("🎉 All done! Find your model and artifacts in the `models/` folder.")


if __name__ == "__main__":
    main()
