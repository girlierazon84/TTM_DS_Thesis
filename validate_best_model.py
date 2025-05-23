#!/usr/bin/env python3
"""
validating_best_model.py

Perform stratified k-fold cross-validation on your saved pipeline,
report mean±std for accuracy, precision, recall, F1; produce a
classification report (decoded labels); and save a boxplot of the
fold‐by‐fold scores.

Usage:
    python validating_best_model.py
"""

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report

# ─── PATHS & SETTINGS ──────────────────────────────────────────────────────────
DB_PATH      = "database/ttm_database.db"
MODEL_DIR    = "best_model"
BEST_MODEL   = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCOD  = os.path.join(MODEL_DIR, "label_encoder.pkl")
CV_PLOT      = os.path.join(MODEL_DIR, "cv_metrics.png")

N_FOLDS      = 5
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")


def load_and_merge_all():
    """Rebuild the same DataFrame you trained on (demo + beh + count_ones_*)."""
    conn = sqlite3.connect(DB_PATH)

    # demographics
    demo = pd.read_sql("SELECT * FROM demographics", conn)
    if {"age", "age_of_onset"}.issubset(demo.columns):
        demo["years_since_onset"] = demo["age"] - demo["age_of_onset"]

    # behaviour patterns
    beh = pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns", conn)
    if "pulling_frequency" in beh.columns:
        fmap = {"Daily":5,"Several times a week":4,"Weekly":3,"Monthly":2,"Rarely":1}
        beh["pulling_frequency_encoded"] = beh["pulling_frequency"].map(fmap).fillna(0).astype(int)
    if "pulling_awareness" in beh.columns:
        amap = {"Yes":1.0,"Sometimes":0.5,"No":0.0}
        beh["awareness_level_encoded"] = beh["pulling_awareness"].map(amap).fillna(0.0)

    # any other 0/1 tables → count_ones_<table>
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
         WHERE type='table'
           AND name NOT IN ('demographics','hair_pulling_behaviours_patterns')
    """)
    tables = [r[0] for r in cur.fetchall()]

    count_dfs = []
    for tbl in tables:
        dfb = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        if "id" not in dfb.columns:
            continue
        bin_cols = [c for c in dfb.columns
                    if c!="id" and set(dfb[c].dropna().unique()) <= {0,1}]
        if not bin_cols:
            continue
        cnt = dfb[bin_cols].sum(axis=1)
        cnt.name = f"count_ones_{tbl}"
        count_dfs.append(pd.DataFrame({"id": dfb["id"], cnt.name: cnt}).set_index("id"))

    conn.close()

    # merge
    X = demo.set_index("id").join(beh.set_index("id"), how="inner")
    for cdf in count_dfs:
        X = X.join(cdf, how="left")
    X = X.fillna(0).reset_index()
    return X


def prepare_X_y(df):
    """
    Split out X/y, load & apply your saved LabelEncoder,
    drop raw text cols the pipeline didn’t expect, and
    **cast all object columns to str** so OHE sees uniform types.
    """
    # target
    raw_y = df["relapse_risk_tag"].astype(str)
    le    = joblib.load(LABEL_ENCOD)
    y     = le.transform(raw_y)

    # drop unused
    X = df.drop(columns=[
        "id","relapse_risk_tag",
        "pulling_frequency","pulling_awareness"
    ], errors="ignore")

    # force all remaining object columns to strings
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)

    return X, y, le


def main():
    print("🔍 Rebuilding full feature-frame from DB…")
    df_all = load_and_merge_all()

    print("⚙️  Preparing X and y…")
    X, y, le = prepare_X_y(df_all)

    print(f"📦 Loading pipeline from {BEST_MODEL}…")
    pipe = joblib.load(BEST_MODEL)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy","precision_macro","recall_macro","f1_macro"]

    print(f"🔁 Running {N_FOLDS}-fold cross-validation…")
    results = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise"
    )

    # report mean ± std
    for metric in scoring:
        scores = results[f"test_{metric}"]
        print(f"• {metric:16s}: {scores.mean():.3f} ± {scores.std():.3f}")

    # classification report via cross-val predictions
    print("\n📝 Classification Report (cross-val):")
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    y_true_lbl = le.inverse_transform(y)
    y_pred_lbl = le.inverse_transform(y_pred)
    print(classification_report(y_true_lbl, y_pred_lbl, digits=3))

    # boxplot of fold scores
    df_scores = pd.DataFrame({
        m: results[f"test_{m}"] for m in scoring
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_scores, ax=ax)
    ax.set_title(f"{N_FOLDS}-Fold CV Metrics")
    ax.set_ylabel("Score")
    plt.tight_layout()
    fig.savefig(CV_PLOT, dpi=150)
    plt.close(fig)

    print(f"\n✅ CV boxplot saved to {CV_PLOT}")


if __name__ == "__main__":
    main()
