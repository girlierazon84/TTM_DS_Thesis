#!/usr/bin/env python3
# testing_best_model.py

"""
Load the best-trained pipeline and held-out test set, compute classification
metrics on the original label names, and save a confusion matrix figure.

Usage:
    python testing_best_model.py
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ── Paths ────────────────────────────────────────────────────────────────────────
MODEL_DIR       = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models"
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENC_FILE  = os.path.join(MODEL_DIR, "label_encoder.pkl")
TEST_FEAT_FILE  = os.path.join(MODEL_DIR, "test_features.csv")
TEST_TARG_FILE  = os.path.join(MODEL_DIR, "test_target.csv")
FIG_DIR         = MODEL_DIR  # where to save confusion matrix
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    """Test the best model on the held-out set, report metrics, and save a confusion matrix."""
    # 1) Load model & encoder
    pipeline = joblib.load(BEST_MODEL_FILE)
    label_enc = joblib.load(LABEL_ENC_FILE)

    # 2) Load test data
    X_test = pd.read_csv(TEST_FEAT_FILE)
    y_numeric = pd.read_csv(TEST_TARG_FILE)["target"].values

    # 3) Predict numeric labels
    y_pred_numeric = pipeline.predict(X_test)

    # 4) Decode to original string labels
    y_test = label_enc.inverse_transform(y_numeric)
    y_pred = label_enc.inverse_transform(y_pred_numeric)

    # 5) Classification report
    print("\nClassification Report (decoded labels):")
    print(classification_report(y_test, y_pred, digits=3))

    # 6) Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_enc.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_enc.classes_
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 7) Save figure
    cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"✅ Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
# ── End of script ────────────────────────────────────────────────────────────────
