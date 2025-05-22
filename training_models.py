#!/usr/bin/env python3
# training_models.py

"""
Train & select the best model to predict relapse risk.

This script:
 1. Reads from your TTM SQLite database:
      - demographics
      - hair_pulling_behaviours_patterns
      - any number of *_1_yes_0_no tables
 2. Preprocesses each:
      • demographics       → add years_since_onset
      • behaviour patterns → encode frequency & awareness, tag relapse_risk_tag
      • each binary table  → sum per-row Yes counts
 3. Merges them on `id`
 4. Splits into train/test, trains four classifiers, picks the best
 5. Serializes:
      • models/best_model.pkl
      • models/label_encoder.pkl
      • models/test_features.csv
      • models/test_target.csv

Usage:
    python training_models.py
"""

import os
import sqlite3
from functools import reduce
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

# ── Paths & constants ─────────────────────────────────────────────────────────
DB_PATH         = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
MODEL_DIR       = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models"
TEST_FEAT_FILE  = os.path.join(MODEL_DIR, "test_features.csv")
TEST_TARG_FILE  = os.path.join(MODEL_DIR, "test_target.csv")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENC_FILE  = os.path.join(MODEL_DIR, "label_encoder.pkl")

RANDOM_STATE    = 42
TEST_SIZE       = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_merge(db_path: str) -> pd.DataFrame:
    """
    Load & preprocess each table, then merge on 'id'.

    Returns
    -------
    merged_df : pd.DataFrame
    """
    conn = sqlite3.connect(db_path)
    # — 1) demographics —
    df_demo = pd.read_sql_query("SELECT * FROM demographics", conn)
    if {"age","age_of_onset"}.issubset(df_demo.columns):
        df_demo["years_since_onset"] = df_demo["age"] - df_demo["age_of_onset"]

    # — 2) behaviour patterns —
    df_beh = pd.read_sql_query(
        "SELECT * FROM hair_pulling_behaviours_patterns", conn)
    # encode frequency
    if "pulling_frequency" in df_beh:
        freq_map = {"Daily":5, "Several times a week":4,
                    "Weekly":3, "Monthly":2, "Rarely":1}
        df_beh["pulling_frequency_encoded"] = (
            df_beh["pulling_frequency"]
            .map(freq_map).fillna(0).astype(int))
    # encode awareness
    if "pulling_awareness" in df_beh:
        aw_map = {"Yes":1.0, "Sometimes":0.5, "No":0.0}
        df_beh["awareness_level_encoded"] = (
            df_beh["pulling_awareness"]
            .map(aw_map).fillna(0.0))
    # tag relapse risk
    def tag_risk(r):
        sev = r.get("pulling_severity",0)
        aw  = r.get("awareness_level_encoded",0)
        if sev>=7 and aw<=0.5:     return "high"
        elif sev>=5:               return "moderate"
        else:                      return "low"
    df_beh["relapse_risk_tag"] = df_beh.apply(tag_risk, axis=1)

    # — 3) binary tables → sum Yes per row —
    cur = conn.cursor()
    cur.execute("""
      SELECT name FROM sqlite_master
       WHERE type='table' AND name LIKE '%\\_1_yes\\_0_no'
      """)
    bin_tables = [r[0] for r in cur.fetchall()]
    counts_dfs = []
    for tbl in bin_tables:
        dfb = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
        if "id" not in dfb:
            continue
        cols = [c for c in dfb.columns if c!="id"]
        # coerce numeric, sum
        dfb[cols] = (dfb[cols]
                     .apply(pd.to_numeric,errors="coerce")
                     .fillna(0).astype(int))
        counts = dfb[cols].sum(axis=1)
        counts_dfs.append(
            pd.DataFrame({
              "id": dfb["id"],
              f"{tbl}_yes_count": counts
            })
        )

    conn.close()

    # — merge all together —
    dfs = [df_demo, df_beh] + counts_dfs
    merged = reduce(lambda a, b: pd.merge(a, b, on="id", how="inner", validate="one_to_one"), dfs)
    return merged


def prepare_features(df: pd.DataFrame, target: str = "relapse_risk_tag"):
    """
    Split into X (numeric features) and y (encoded target).

    Returns
    -------
    X : pd.DataFrame
    y : np.ndarray
    """
    if target not in df:
        raise KeyError(f"Target '{target}' missing")
    y = df[target].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LABEL_ENC_FILE)

    X = (df
         .drop(columns=["id", target])
         .select_dtypes(include=[np.number])
         .copy())
    return X, y_enc


def train_and_select(X_train, y_train, X_test, y_test):
    """
    Train multiple pipelines, pick best by accuracy.
    """
    models = {
      "LogisticRegression":
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
      "RandomForest":
        RandomForestClassifier(n_estimators=100, max_depth=10,
                               max_features="sqrt", min_samples_leaf=1, random_state=RANDOM_STATE),
      "GradientBoost":
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                   random_state=RANDOM_STATE),
      "MLPClassifier":
        MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500,
                      random_state=RANDOM_STATE),
    }

    best_acc, best_name, best_pipe = -np.inf, None, None

    for name, clf in models.items():
        pipe = Pipeline([
          ("scaler", StandardScaler()),
          ("clf",    clf)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:<15s} test accuracy: {acc:.3f}")
        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    print(f"\n✅ Best model: {best_name} (accuracy={best_acc:.3f})\n")
    return best_name, best_pipe


def main():
    """Orchestrate loading, training, and serialization."""
    print("🔍 Loading & merging data…")
    df = load_and_merge(DB_PATH)

    print("⚙️ Preparing features & target…")
    X, y = prepare_features(df)

    print(f"🔀 Splitting train/test (test_size={TEST_SIZE})…")
    X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=TEST_SIZE,
      stratify=y,
      random_state=RANDOM_STATE
    )
    X_test.to_csv(TEST_FEAT_FILE, index=False)
    pd.Series(y_test, name="target").to_csv(TEST_TARG_FILE, index=False)

    print("🏋️ Training & selecting best model…")
    _, best_pipe = train_and_select(X_train, y_train, X_test, y_test)

    print(f"💾 Saving best pipeline to {BEST_MODEL_FILE}…")
    joblib.dump(best_pipe, BEST_MODEL_FILE)

    print("🎉 Finished. Artifacts:")
    print("  -", BEST_MODEL_FILE)
    print("  -", LABEL_ENC_FILE)
    print("  -", TEST_FEAT_FILE)
    print("  -", TEST_TARG_FILE)


if __name__ == "__main__":
    main()
