"""
validating_best_model.py

Perform k-fold cross-validation on the best pipeline loaded from disk,
producing summary statistics and a boxplot of fold scores.

Usage:
    python validating_best_model.py
"""

import os
import sqlite3
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# --- Paths ---
DB_PATH         = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
TABLE_NAME      = "trichotillomania_data"         # name of the table containing preprocessed data
BEST_MODEL_FILE = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models\best_model.pkl"
LABEL_ENC_FILE  = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models\label_encoder.pkl"
FIG_DIR         = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models"
CV_PLOT_FILE    = os.path.join(FIG_DIR, "cv_scores_boxplot.png")

# --- Constants ---
RANDOM_STATE    = 42
FOLDS           = 5

# --- Create directories if they don't exist ---
os.makedirs(FIG_DIR, exist_ok=True)


def load_data_from_db(db_path: str, table: str) -> pd.DataFrame:
    """
    Load the preprocessed survey_responses table from SQLite.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (.db) file.
    table : str
        Name of the table containing preprocessed data.

    Returns
    -------
    pd.DataFrame
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def prepare_X_y(df: pd.DataFrame, target_col: str = "relapse_risk_tag"):
    """
    Extract numeric features and encode the categorical target.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame loaded from SQLite.
    target_col : str
        Name of the target column (categorical).

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix.
    y : np.ndarray
        Encoded target array.
    """
    # load label encoder that was saved alongside the model
    le = joblib.load(LABEL_ENC_FILE)

    y_raw = df[target_col].astype(str)
    y = le.transform(y_raw)

    # select only numeric columns for X
    X = df.select_dtypes(include=["number"]).copy()

    return X, y


def main():
    """The main function to run the script."""
    # 1. Load preprocessed data directly from SQLite
    print("🔍 Loading preprocessed data from database…")
    df = load_data_from_db(DB_PATH, TABLE_NAME)

    # 2. Prepare feature matrix X and encoded target y
    print("⚙️  Preparing X and y …")
    X, y = prepare_X_y(df)

    # 3. Load the saved best model pipeline
    print(f"📦 Loading best model from {BEST_MODEL_FILE} …")
    pipe = joblib.load(BEST_MODEL_FILE)

    # 4. k-fold cross-validation
    print(f"🔄 Performing {FOLDS}-fold cross-validation…")
    scores = cross_val_score(
        pipe, X, y,
        cv=FOLDS,
        scoring="accuracy",
        n_jobs=-1
    )

    print(f"▶ Fold accuracies: {scores}")
    print(f"▶ Mean accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

    # 5. Plot boxplot of CV scores
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(scores, patch_artist=True)
    ax.set_title(f"{FOLDS}-Fold CV Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([1])
    ax.set_xticklabels([f"{FOLDS}-fold"])
    fig.tight_layout()
    fig.savefig(CV_PLOT_FILE, dpi=150)
    plt.close(fig)

    print(f"✅ CV boxplot saved to {CV_PLOT_FILE}")


if __name__ == "__main__":
    main()
# --- End of script ---
