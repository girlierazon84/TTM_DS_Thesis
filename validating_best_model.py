#!/usr/bin/env python3
"""
validating_best_model.py

Load your raw tables from SQLite, engineer exactly the same features
you used in training, then perform stratified k-fold cross-validation
on the saved pipeline.

Usage:
    python validating_best_model.py
"""
import os
import sqlite3
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Config ──────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
DB_PATH          = os.path.join(BASE_DIR, "database", "ttm_database.db")
MODEL_DIR        = os.path.join(BASE_DIR, "models")
BEST_MODEL_FILE  = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENC_FILE   = os.path.join(MODEL_DIR, "label_encoder.pkl")
FIG_DIR          = os.path.join(MODEL_DIR, "figures")
CV_PLOT_FILENAME = "cv_accuracy_boxplot.png"

RANDOM_STATE = 42
FOLDS        = 5

os.makedirs(FIG_DIR, exist_ok=True)


def load_and_merge(db_path: str) -> pd.DataFrame:
    """
    Load raw tables from SQLite, compute features and binary-counts,
    merge into one DataFrame keyed by `id`.
    """
    conn = sqlite3.connect(db_path)

    # 1) demographics
    demo = pd.read_sql_query("SELECT * FROM demographics", conn, parse_dates=["timestamp"])
    if {"age", "age_of_onset"}.issubset(demo.columns):
        demo["years_since_onset"] = demo["age"] - demo["age_of_onset"]
    else:
        demo["years_since_onset"] = 0

    # 2) hair pulling behaviours
    beh = pd.read_sql_query("SELECT * FROM hair_pulling_behaviours_patterns", conn, parse_dates=["timestamp"])
    # frequency encoding
    if "pulling_frequency" in beh.columns:
        freq_map = {"Daily":5, "Several times a week":4, "Weekly":3, "Monthly":2, "Rarely":1}
        beh["pulling_frequency_encoded"] = beh["pulling_frequency"].map(freq_map).fillna(0).astype(int)
    else:
        beh["pulling_frequency_encoded"] = 0
    # awareness encoding
    if "pulling_awareness" in beh.columns:
        aw_map = {"Yes":1.0, "Sometimes":0.5, "No":0.0}
        beh["awareness_level_encoded"] = beh["pulling_awareness"].map(aw_map).fillna(0.0)
    else:
        beh["awareness_level_encoded"] = 0.0
    # relapse risk tag
    def _tag(r):
        sev = r.get("pulling_severity", 0)
        aw  = r.get("awareness_level_encoded", 0)
        if sev >= 7 and aw <= 0.5: return "high"
        if sev >= 5:               return "moderate"
        return "low"
    beh["relapse_risk_tag"] = beh.apply(_tag, axis=1)

    # 3) binary tables → yes-counts per id
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name GLOB '*_1_yes_0_no'
    """)
    binary_tables = [row[0] for row in cur.fetchall()]

    counts = []
    for tbl in binary_tables:
        dfb = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
        if "id" not in dfb.columns:
            continue
        # drop any precomputed columns
        features = [c for c in dfb.columns if c != "id"]
        # sum 1s
        yes_sum = dfb[features].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).sum(axis=1)
        counts.append(pd.DataFrame({
            "id": dfb["id"],
            # sanitize the feature-name for the column
            f"{tbl}_yes_count": yes_sum
        }))

    conn.close()

    # 4) merge all
    df = demo.set_index("id")\
           .join(beh.set_index("id"), how="inner", rsuffix="_beh", on="id", validate="one_to_one")\
           .reset_index()
    for cts in counts:
        df = df.merge(cts, on="id", how="left").fillna(0)

    return df


def prepare_X_y(df: pd.DataFrame, target_col: str = "relapse_risk_tag"):
    """
    Split DataFrame into numeric X and encoded y.
    """
    # load the exact LabelEncoder used at training
    le = joblib.load(LABEL_ENC_FILE)

    y = le.transform(df[target_col].astype(str))
    X = df.select_dtypes(include="number").copy()
    return X, y


def main():
    """Function to run the script."""
    print("🔍 Loading & merging raw tables…")
    df = load_and_merge(DB_PATH)

    print("⚙️ Preparing X & y…")
    X, y = prepare_X_y(df)

    print(f"📦 Loading pipeline from {BEST_MODEL_FILE}…")
    pipe = joblib.load(BEST_MODEL_FILE)

    print(f"🔄 Running {FOLDS}-fold stratified CV…")
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print("\n▶ Fold accuracies:", scores)
    print(f"▶ Mean = {scores.mean():.3f} ± {scores.std():.3f}\n")

    # boxplot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(scores, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue"),
               medianprops=dict(color="darkblue"))
    ax.set_title(f"{FOLDS}-Fold CV Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([1])
    ax.set_xticklabels([f"{FOLDS}-fold"])
    plt.tight_layout()

    out_file = os.path.join(FIG_DIR, CV_PLOT_FILENAME)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"✅ Saved CV boxplot to {out_file}")


if __name__ == "__main__":
    main()
