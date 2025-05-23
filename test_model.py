#!/usr/bin/env python3
"""
testing_best_model.py

Load the best-trained pipeline and test on the 20% hold-out set.
Print a classification report (with decoded labels) and save
a confusion matrix figure.

Usage:
    python testing_best_model.py
"""

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ─── PATHS & CONSTANTS ─────────────────────────────────────────────────────────
DB_PATH       = "database/ttm_database.db"
MODEL_DIR     = "best_model"
BEST_MODEL    = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCOD   = os.path.join(MODEL_DIR, "label_encoder.pkl")
CM_PNG        = os.path.join(MODEL_DIR, "confusion_matrix.png")
SEED          = 42
TEST_SIZE     = 0.2


def load_and_merge_all():
    """Reproduce the exact feature‐engineering from training_models.py."""
    conn = sqlite3.connect(DB_PATH)

    # demographics
    demo = pd.read_sql("SELECT * FROM demographics", conn)
    if {"age", "age_of_onset"}.issubset(demo.columns):
        demo["years_since_onset"] = demo["age"] - demo["age_of_onset"]

    # behaviour
    beh = pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns", conn)
    if "pulling_frequency" in beh.columns:
        freq_map = {
            "Daily":5, "Several times a week":4,
            "Weekly":3, "Monthly":2, "Rarely":1
        }
        beh["pulling_frequency_encoded"] = beh["pulling_frequency"].map(freq_map).fillna(0).astype(int)
    if "pulling_awareness" in beh.columns:
        aw_map = {"Yes":1.0, "Sometimes":0.5, "No":0.0}
        beh["awareness_level_encoded"] = beh["pulling_awareness"].map(aw_map).fillna(0.0)

    # any other binary tables → count_ones
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
         WHERE type='table'
           AND name NOT IN ('demographics','hair_pulling_behaviours_patterns')
    """)
    others = [r[0] for r in cur.fetchall()]

    bin_dfs = []
    for tbl in others:
        dfb = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        if "id" not in dfb.columns:
            continue
        bin_cols = [
            c for c in dfb.columns if c!="id"
            and set(dfb[c].dropna().unique()).issubset({0,1})
        ]
        if not bin_cols:
            continue
        dfb["count_ones_"+tbl] = dfb[bin_cols].sum(axis=1)
        bin_dfs.append(dfb[["id","count_ones_"+tbl]].set_index("id"))

    conn.close()

    X = demo.set_index("id").join(beh.set_index("id"), how="inner")
    for bdf in bin_dfs:
        X = X.join(bdf, how="left")
    return X.fillna(0).reset_index()


def prepare_X_y(df):
    """Split out X/y and label-encode relapse_risk_tag."""
    y = df["relapse_risk_tag"].astype(str)
    le = joblib.load(LABEL_ENCOD)  # use the same encoder from training
    y_enc = le.transform(y)

    # drop columns not used by the final pipeline
    X = df.drop(columns=[
        "id", "relapse_risk_tag", "pulling_frequency", "pulling_awareness"
    ], errors="ignore")
    return X, y_enc, le


def main():
    print("🔍 Rebuilding feature matrix…")
    df_all = load_and_merge_all()

    print("⚙️ Splitting into train/test…")
    X, y, le = prepare_X_y(df_all)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    print(f"📦 Loading best pipeline from {BEST_MODEL}…")
    pipe = joblib.load(BEST_MODEL)

    print("🧮 Evaluating on hold-out set…")
    y_pred = pipe.predict(X_test)

    # decode back to string labels
    y_true_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print("\nClassification Report (decoded labels):")
    print(classification_report(y_true_labels, y_pred_labels, digits=3))

    # confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)

    print(f"\n✅ Confusion matrix saved to {CM_PNG}\n")


if __name__ == "__main__":
    main()
