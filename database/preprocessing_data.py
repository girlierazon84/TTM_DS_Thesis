#!/usr/bin/env python3
"""
preprocessing_data.py

Load raw survey data from your SQLite database, clean & engineer features,
tag relapse risk, and save the cleaned version back into the same DB.

Functions
---------
- load_data_from_db(db_path: str, table: str) -> pd.DataFrame
    Load the raw survey table into a DataFrame (parsing any `timestamp`).
- preprocess_data(df: pd.DataFrame) -> pd.DataFrame
    Clean, handle missing values, engineer features, encode categoricals,
    compute counts, one-hot gender, and tag relapse risk — all safely
    (if a column is missing, it just fills a default).
- save_to_db(df: pd.DataFrame, db_path: str, table: str) -> None
    Write the cleaned DataFrame into a new SQLite table (replacing if exists).

Usage
-----
    python preprocessing_data.py
"""
import sqlite3
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────────
DB_PATH     = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
RAW_TABLE   = "trichotillomania_data"    # or your merged/raw table name
CLEAN_TABLE = "preprocessed_ttm_data"      # new table for downstream EDA/modeling

# ── 1. LOAD RAW DATA ────────────────────────────────────────────────────────────
def load_data_from_db(db_path: str, table: str) -> pd.DataFrame:
    """
    Connect to SQLite and load all rows from `table`, parsing any `timestamp` columns.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    # parse any timestamp-like columns
    for col in df.columns:
        if "timestamp" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ── 2. PREPROCESS ────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features.

    Missing columns are handled gracefully by filling sensible defaults.
    """
    df = df.copy()

    # 1. Drop rows missing core essentials (if present)
    essentials = ["age", "age_of_onset", "pulling_severity",
                  "pulling_frequency", "pulling_awareness"]
    existing_essentials = [c for c in essentials if c in df.columns]
    if existing_essentials:
        df.dropna(subset=existing_essentials, inplace=True)

    # 2. years since onset
    if "age" in df.columns and "age_of_onset" in df.columns:
        df["years_since_onset"] = df["age"] - df["age_of_onset"]
    else:
        df["years_since_onset"] = 0

    # 3. encode mappings (only if source col exists)
    def safe_map(src, dst, mapping, dtype=int):
        if src in df.columns:
            df[dst] = df[src].map(mapping).fillna(0).astype(dtype)
        else:
            df[dst] = 0

    safe_map("pulling_frequency", "pulling_frequency_encoded",
             {"Daily":5,"Several times a week":4,"Weekly":3,"Monthly":2,"Rarely":1})
    safe_map("pulling_awareness", "awareness_level_encoded",
             {"Yes":1.0,"Sometimes":0.5,"No":0.0}, dtype=float)
    safe_map("seasonal_change", "seasonal_change_binary", {"Yes":1,"No":0})
    safe_map("support_sought",  "support_sought_binary",  {"Yes":1,"No":0})
    safe_map("therapy_sought",  "therapy_sought_binary",  {"Yes":1,"No":0})
    safe_map("successfully_stopped","stopped_binary",    {"Yes":1,"No":0})

    # 4. text-derived counts
    def safe_count(src, dst):
        if src in df.columns:
            df[dst] = df[src].fillna("").apply(lambda x: len(x.split(",")))
        else:
            df[dst] = 0

    safe_count("emotions_before_pulling",       "emotional_trigger_score")
    safe_count("coping_strategies",             "coping_strategies_count")
    safe_count("other_mental_conditions",       "mental_health_condition_count")
    safe_count("activities_lists",              "activities_count")
    safe_count("seasons_affected",              "seasons_affected_count")

    # 5. one-hot encode gender
    if "gender" in df.columns:
        df = pd.get_dummies(df, columns=["gender"], prefix="gender")

    # 6. tag relapse risk
    def tag_risk(row):
        sev = row.get("pulling_severity", 0)
        aw  = row.get("awareness_level_encoded", 0)
        if sev >= 7 and aw <= 0.5:
            return "high"
        elif sev >= 5:
            return "moderate"
        else:
            return "low"
    df["relapse_risk_tag"] = df.apply(tag_risk, axis=1)

    # 7. finalize
    df.reset_index(drop=True, inplace=True)
    return df

# ── 3. SAVE CLEANED DATA ─────────────────────────────────────────────────────────
def save_to_db(df: pd.DataFrame, db_path: str, table: str) -> None:
    """
    Write DataFrame into SQLite as `table` (replacing if exists).
    """
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Saved cleaned data to [{table}] in {db_path}")

# ── RUN AS SCRIPT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw   = load_data_from_db(DB_PATH, RAW_TABLE)
    clean = preprocess_data(raw)
    save_to_db(clean, DB_PATH, CLEAN_TABLE)
    print("✅ Preprocessing complete — ready for EDA.")
