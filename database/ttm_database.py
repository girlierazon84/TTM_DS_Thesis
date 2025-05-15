"""This script creates a SQLite database from a semicolon-delimited CSV file."""

import sqlite3
import pandas as pd


# === Paths ===
CSV_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\survey_responses_cleaned.csv"
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"

# === Load CSV with semicolon separator ===
df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')

# === Connect to SQLite ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# === Drop existing table (optional) ===
cursor.execute("DROP TABLE IF EXISTS trichotillomania_data")

# === Write to database ===
df.to_sql("trichotillomania_data", conn, if_exists="replace", index=False)

# === Finalize ===
conn.commit()
conn.close()

print("✅ SQLite TTM database created successfully!")
