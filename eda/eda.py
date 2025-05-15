"""
Exploratory Data Analysis (EDA) and Visualization
- Summarizes key variables for thesis and app
- Saves plots to PNGs
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Setup ===
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\ttm_research_study\database\survey_responses.db"
OUTPUT_DIR = r"C:\Users\girli\OneDrive\Desktop\ttm_research_study\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM survey_responses", conn)
conn.close()

# === EDA Plots ===

# Plot 1: Severity distribution
plt.figure(figsize=(6,4))
sns.histplot(df["pulling_severity"], bins=10, kde=True)
plt.title("Pulling Severity Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "severity_distribution.png"))

# Plot 2: Severity by gender
plt.figure(figsize=(6,4))
sns.boxplot(x="gender", y="pulling_severity", data=df)
plt.title("Severity by Gender")
plt.savefig(os.path.join(OUTPUT_DIR, "severity_by_gender.png"))

# Plot 3: Seasonal pattern
plt.figure(figsize=(6,4))
sns.countplot(x="seasonal_change", hue="relapse_risk_tag", data=df)
plt.title("Seasonal Change vs Relapse Risk")
plt.savefig(os.path.join(OUTPUT_DIR, "seasonal_vs_relapse.png"))
