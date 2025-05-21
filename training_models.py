"""
training_models.py

Load preprocessed TTM survey data from SQLite, train multiple classification models
to predict relapse risk, evaluate them on a held-out test set, and save the best‐
performing pipeline to disk for later use.

Models trained:
 - Logistic Regression
 - Random Forest
 - Gradient Boosting
 - Multi-layer Perceptron

Outputs:
 - models/best_model.pkl       : serialized best pipeline
 - models/label_encoder.pkl    : target label encoder
 - models/test_features.csv    : held-out test features
 - models/test_target.csv      : held-out test labels

Usage:
    python training_models.py
"""

import os
import sqlite3
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

# --- Paths & constants ---
DB_PATH         = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
TABLE_NAME      = "trichotillomania_data"
MODEL_DIR       = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\models"
TEST_FEAT_FILE  = os.path.join(MODEL_DIR, "test_features.csv")
TEST_TARG_FILE  = os.path.join(MODEL_DIR, "test_target.csv")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENC_FILE  = os.path.join(MODEL_DIR, "label_encoder.pkl")
# Constants for reproducibility
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(db_path: str, table: str) -> pd.DataFrame:
    """
    Load the cleaned survey_responses table from a SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite .db file.
    table : str
        Name of the table containing the processed responses.

    Returns
    -------
    pd.DataFrame
        DataFrame of all rows in the table.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def prepare_features(df: pd.DataFrame, target_col: str = "relapse_risk_tag"):
    """
    Separate features X from target y, encode the target, and select numeric X.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed DataFrame.
    target_col : str
        Name of the categorical target column.

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix.
    y : np.ndarray
        Encoded target array.
    """
    # encode target
    y = df[target_col].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # persist the label mapping
    joblib.dump(le, LABEL_ENC_FILE)

    # select numeric predictors
    X = df.select_dtypes(include=[np.number]).copy()
    return X, y_enc


def train_and_select(X_train, y_train, X_test, y_test):
    """
    Train multiple pipelines, evaluate on the test set,
    and return the best one by accuracy.

    Returns
    -------
    best_name : str
    best_pipeline : sklearn.Pipeline
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest"     : RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            min_samples_leaf=1,
            max_features="sqrt",
            max_depth=10,
        ),
        "GradientBoost"    : GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE
            ),
        "MLPClassifier"    : MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, random_state=RANDOM_STATE),
    }

    best_acc = -np.inf
    best_name, best_pipe = None, None

    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",     clf)
        ], memory=None)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:15s} test accuracy: {acc:.3f}")
        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    print(f"\n✅ Best model: {best_name} (accuracy={best_acc:.3f})\n")
    return best_name, best_pipe


def main():
    """Main function to load data, train models, and save the best one."""
    print("🔍 Loading data from SQLite…")
    df = load_data(DB_PATH, TABLE_NAME)

    print("⚙️  Preparing features and target…")
    X, y = prepare_features(df)

    print(f"🔀 Splitting into train/test (test_size={TEST_SIZE})…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # persist the held-out test set for downstream evaluation
    X_test.to_csv(TEST_FEAT_FILE, index=False)
    pd.Series(y_test, name="target").to_csv(TEST_TARG_FILE, index=False)

    print("🏋️  Training models and selecting best…")
    _, best_pipeline = train_and_select(X_train, y_train, X_test, y_test)

    print(f"💾 Saving best model to {BEST_MODEL_FILE}…")
    joblib.dump(best_pipeline, BEST_MODEL_FILE)

    print("🎉 Done! Your model and artifacts are in:")
    print("  -", BEST_MODEL_FILE)
    print("  -", LABEL_ENC_FILE)
    print("  -", TEST_FEAT_FILE)
    print("  -", TEST_TARG_FILE)


if __name__ == "__main__":
    main()
# --- End of file ---
