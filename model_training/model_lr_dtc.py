"""
Model Training and Evaluation
This script trains and evaluates two models: Logistic Regression and Decision Tree Classifier.
"""

import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# === Paths ===
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
PLOTS_DIR = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\images\models_img"
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Load Data ===
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM trichotillomania_data", conn)
conn.close()

# === Feature Engineering ===
# Drop rows with missing relapse_risk_tag
df = df.dropna(subset=["relapse_risk_tag"])

# Select features for modeling
features = [
    "age", "age_of_onset", "pulling_severity",
    "gender", "pulling_frequency", "pulling_awareness",
    "coping_strategies", "other_mental_conditions"
]
df = df[features + ["relapse_risk_tag"]].copy()

# Encode categorical features
label_encoders = {}
for col in ["gender", "pulling_frequency", "pulling_awareness"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Count-based features
df["coping_strategies_count"] = df["coping_strategies"].fillna("").apply(lambda x: len(x.split(",")))
df["mental_health_condition_count"] = df["other_mental_conditions"].fillna("").apply(lambda x: len(x.split(",")))
df = df.drop(columns=["coping_strategies", "other_mental_conditions"])

# Encode target
target_le = LabelEncoder()
df["relapse_risk_tag_encoded"] = target_le.fit_transform(df["relapse_risk_tag"])
df = df.drop(columns=["relapse_risk_tag"])

# === Train/Test Split ===
X = df.drop(columns=["relapse_risk_tag_encoded"])
y = df["relapse_risk_tag_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Model Training ===
# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# === Evaluation ===
report_lr = classification_report(
    y_test, y_pred_lr, target_names=target_le.classes_, output_dict=True
    )
report_dt = classification_report(
    y_test, y_pred_dt, target_names=target_le.classes_, output_dict=True
    )

# === Confusion Matrices ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", ax=axs[0])
axs[0].set_title("Logistic Regression")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", ax=axs[1])
axs[1].set_title("Decision Tree")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"))
plt.close()

# === Feature Importance ===
importance = pd.Series(dt.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance (Decision Tree)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
plt.close()

# Save processed data
PROCESSED_CSV = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\survey_processed_for_modeling.csv"
df.to_csv(PROCESSED_CSV, index=False)
