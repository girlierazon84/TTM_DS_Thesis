#!/usr/bin/env python3
"""
validating_best_model.py

Load best pipeline and run k-fold CV directly on your merged table,
plot boxplot of fold accuracies, and save to PNG.
"""
import os
import sqlite3
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

DB_PATH    = "database/ttm_database.db"
BEST_MODEL = "models/best_model.pkl"
LABEL_ENC  = "models/label_encoder.pkl"
FIG_OUT    = "models/cv_boxplot.png"
FOLDS      = 5
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

def load_Xy():
    conn = sqlite3.connect(DB_PATH)
    demo = pd.read_sql("SELECT * FROM demographics",conn).set_index("id")
    beh  = pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns",conn).set_index("id")
    conn.close()
    df   = demo.join(beh, how="inner", on="id", validate="one_to_one").dropna(subset=["relapse_risk_tag"])
    le   = joblib.load(LABEL_ENC)
    y    = le.transform(df["relapse_risk_tag"].astype(str))
    X    = df.select_dtypes(include=["number"]).fillna(0)
    return X,y

def main():
    X,y = load_Xy()
    pipe = joblib.load(BEST_MODEL)
    scores = cross_val_score(pipe, X, y, cv=FOLDS, scoring="accuracy", n_jobs=-1)
    print("Scores:", scores, f"Mean={scores.mean():.3f}±{scores.std():.3f}")
    fig,ax = plt.subplots(figsize=(6,4))
    ax.boxplot(scores, patch_artist=True)
    ax.set(xticks=[1], xticklabels=[f"{FOLDS}-fold"], ylabel="Accuracy", title="CV Accuracies")
    fig.tight_layout(); fig.savefig(FIG_OUT, dpi=150)
    print(f"Saved CV boxplot to {FIG_OUT}")

if __name__=="__main__":
    main()
