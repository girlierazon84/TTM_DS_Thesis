#!/usr/bin/env python3
"""
training_models.py

Load your SQLite tables, assemble a full feature matrix for relapse-risk prediction,
train multiple pure-scikit-learn classifiers, evaluate them, and save the best.
"""

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ─── PATHS & SEED ──────────────────────────────────────────────────────────────
DB_PATH     = "database/ttm_database.db"
MODEL_DIR   = "best_model"
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_MODEL  = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCOD = os.path.join(MODEL_DIR, "label_encoder.pkl")

SEED = 42
np.random.seed(SEED)


def load_and_merge_all():
    """Load and merge demographics, behaviour, plus binary‐flag tables."""
    conn = sqlite3.connect(DB_PATH)

    # demographics
    demo = pd.read_sql("SELECT * FROM demographics", conn)
    if {"age", "age_of_onset"}.issubset(demo.columns):
        demo["years_since_onset"] = demo["age"] - demo["age_of_onset"]

    # behaviour patterns
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

    # find all other tables
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
        # columns that are strictly 0/1
        bin_cols = [
            c for c in dfb.columns
            if c != "id" and set(dfb[c].dropna().unique()).issubset({0,1})
        ]
        if not bin_cols:
            continue
        # sum them per row
        dfb["count_ones_"+tbl] = dfb[bin_cols].sum(axis=1)
        bin_dfs.append(dfb[["id", "count_ones_"+tbl]].set_index("id"))

    conn.close()

    # left‐join everything on id
    X = demo.set_index("id").join(beh.set_index("id"), how="inner")
    for bdf in bin_dfs:
        X = X.join(bdf, how="left")
    return X.fillna(0).reset_index()


def prepare_X_y(df):
    """Extract X/y and label‐encode the target."""
    y = df["relapse_risk_tag"].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LABEL_ENCOD)

    # drop raw columns we won't feed directly
    X = df.drop(columns=[
        "id", "relapse_risk_tag",
        "pulling_frequency", "pulling_awareness"
    ], errors="ignore")
    return X, y_enc


if __name__ == "__main__":
    print("🔍 Loading & merging tables…")
    df_all = load_and_merge_all()

    print("⚙️ Preparing features & target…")
    X, y = prepare_X_y(df_all)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # ensure all object columns are uniform strings
    for col in X_train.select_dtypes(include=["object", "category"]):
        X_train[col] = X_train[col].astype(str)
        X_test[col]  = X_test[col].astype(str)

    # split numeric vs categorical
    num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # build transformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ])

    # define pure-sklearn pipelines
    pipelines = {
        "LogisticRegression": Pipeline([("prep", preprocessor),
                                        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "RandomForest"      : Pipeline([("prep", preprocessor),
                                        ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED, min_samples_leaf=1, max_features="sqrt"))]),
        "GradientBoosting"  : Pipeline([("prep", preprocessor),
                                        ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED))]),
        "MLPClassifier"     : Pipeline([("prep", preprocessor),
                                        ("clf", MLPClassifier(hidden_layer_sizes=(50,50),
                                                              max_iter=500,
                                                              random_state=SEED))]),
    }

    # train & evaluate
    best_acc, best_name, best_pipe = -1, None, None
    print("🏋️ Training & evaluating…")
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"  • {name:20s} → accuracy = {acc:.3f}")
        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    print(f"\n✅ Best model: {best_name} (accuracy={best_acc:.3f})")
    print("📦 Saving pipeline & encoder…")
    joblib.dump(best_pipe, BEST_MODEL)
    print("🎉 Done! Models saved under `models/`.")
