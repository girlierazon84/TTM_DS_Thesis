"""
Exploratory Data Analysis (EDA) and Visualization Script

- Loads survey responses from a SQLite database
- Generates plots for key variables relevant to the thesis and application
- Saves each plot as a PNG file for reuse in reporting and dashboard
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
OUTPUT_DIR = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM trichotillomania_data", conn)
conn.close()

# === Plot 1: Pulling Severity Distribution ===
plt.figure(figsize=(6, 4))
sns.histplot(df["pulling_severity"], bins=10, kde=True, color='purple')
plt.title("Pulling Severity Distribution")
plt.xlabel("Severity (1-10)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "severity_distribution.png"))

# === Plot 2: Pulling Severity by Gender ===
plt.figure(figsize=(6, 4))
sns.boxplot(x="gender", y="pulling_severity", data=df, palette="Set2")
plt.title("Pulling Severity by Gender")
plt.xlabel("Gender")
plt.ylabel("Severity")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "severity_by_gender.png"))

# === Plot 3: Seasonal Change vs Relapse Risk ===
plt.figure(figsize=(6, 4))
sns.countplot(x="seasonal_change", hue="relapse_risk_tag", data=df, palette="coolwarm")
plt.title("Seasonal Change vs Relapse Risk")
plt.xlabel("Seasonal Change")
plt.ylabel("Number of Respondents")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "seasonal_vs_relapse.png"))

# === Plot 4: Age of Onset Distribution ===
plt.figure(figsize=(6, 4))
sns.histplot(df["age_of_onset"], bins=20, kde=True, color='teal')
plt.title("Age of Onset of Trichotillomania")
plt.xlabel("Age of Onset")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_of_onset_distribution.png"))

# === Plot 5: Age of Onset by Gender ===
plt.figure(figsize=(6, 4))
sns.boxplot(x="gender", y="age_of_onset", data=df, palette="Set3")
plt.title("Age of Onset by Gender")
plt.xlabel("Gender")
plt.ylabel("Age of Onset")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_of_onset_by_gender.png"))

print("✅ EDA plots saved to:", OUTPUT_DIR)
