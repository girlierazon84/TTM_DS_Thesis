"""
testing_best_model.py

Load the best-trained pipeline and held-out test set, compute classification
metrics and save a confusion matrix figure.

Usage:
    python testing_best_model.py
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Paths ---
MODEL_DIR       = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models"
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
TEST_FEAT_FILE  = os.path.join(MODEL_DIR, "test_features.csv")
TEST_TARG_FILE  = os.path.join(MODEL_DIR, "test_target.csv")
FIG_DIR         = MODEL_DIR  # save confusion matrix here

def main():
    """Test the best model on the test set and save confusion matrix."""
    # Load model
    pipe = joblib.load(BEST_MODEL_FILE)

    # Load test data
    X_test = pd.read_csv(TEST_FEAT_FILE)
    y_test = pd.read_csv(TEST_TARG_FILE)["target"]

    # Predict
    preds = pipe.predict(X_test)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"✅ Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
# --- End of script ---
