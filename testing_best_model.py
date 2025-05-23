#!/usr/bin/env python3
"""
testing_best_model.py

Load best_model.pkl and label_encoder.pkl, run on the held-out test split
and output a classification report + confusion matrix PNG.
"""
import os
import joblib
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Paths
MODEL_DIR  = "models"
BEST_MODEL = os.path.join(MODEL_DIR,"best_model.pkl")
LABEL_ENC  = os.path.join(MODEL_DIR,"label_encoder.pkl")
FIG_PATH   = os.path.join(MODEL_DIR,"confusion_matrix.png")
DB_PATH    = "database/ttm_database.db"

def load_test_split():
    conn  = sqlite3.connect(DB_PATH)
    demo  = pd.read_sql("SELECT * FROM demographics",conn).set_index("id")
    beh   = pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns",conn).set_index("id")
    conn.close()
    df    = demo.join(beh, how="inner", on="id", validate="one_to_one")
    df    = df.dropna(subset=["relapse_risk_tag"])
    X     = df.select_dtypes(include=[int,float]).fillna(0)
    y     = df["relapse_risk_tag"].astype(str)
    le    = joblib.load(LABEL_ENC)
    return X, le.transform(y), le

def main():
    pipe, le = joblib.load(BEST_MODEL), joblib.load(LABEL_ENC)
    X,y_true_numeric, le = load_test_split()
    y_pred_numeric = pipe.predict(X)
    y_true = le.inverse_transform(y_true_numeric)
    y_pred = le.inverse_transform(y_pred_numeric)
    print(classification_report(y_true, y_pred, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45)
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved CM to {FIG_PATH}")

if __name__=="__main__":
    main()
