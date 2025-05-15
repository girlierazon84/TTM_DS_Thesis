"""
Train a basic machine learning model for relapse risk classification.
- Saves the trained model to disk.
"""

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

DB_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
MODEL_PATH = "model_random_forest.pkl"

# === Load and preprocess ===
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM trichotillomania_data", conn)
conn.close()

X = df[["pulling_severity", "age", "age_of_onset"]]
y = df["relapse_risk_tag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train ===
model = RandomForestClassifier(
    random_state=42,
    min_samples_leaf=1,
    max_features='sqrt',
    n_estimators=100)
model.fit(X_train, y_train)


joblib.dump(model, MODEL_PATH)
