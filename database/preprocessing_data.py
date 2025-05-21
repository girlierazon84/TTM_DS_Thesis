#!/usr/bin/env python3
"""
preprocessing_data.py

Compute and append new columns directly into your existing tables:

1. demographics:
   - years_since_onset = age - age_of_onset

2. hair_pulling_behaviours_patterns:
   - awareness_level_encoded (map Yes/Sometimes/No → 1.0/0.5/0.0)
   - relapse_risk_tag (high/moderate/low by severity+awareness)

3. All *_1_yes_0_no tables:
   - count_ones (sum of all 1-columns per row)

Each table is replaced in-place in the same SQLite database.
"""
import sqlite3
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────────
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"

# ── 1. DEMOGRAPHICS ─────────────────────────────────────────────────────────────
def process_demographics(conn: sqlite3.Connection):
    """
    Load the `demographics` table, compute `years_since_onset`, then
    overwrite the `demographics` table including the new column.
    """
    df = pd.read_sql_query("SELECT * FROM demographics", conn)
    if {'age','age_of_onset'}.issubset(df.columns):
        df['years_since_onset'] = df['age'] - df['age_of_onset']
    else:
        df['years_since_onset'] = pd.NA

    df.to_sql('demographics', conn, if_exists='replace', index=False)
    print(f"✅ Updated 'demographics' with years_since_onset ({len(df)} rows)")


# ── 2. BEHAVIOUR PATTERNS ────────────────────────────────────────────────────────
def process_behaviour_patterns(conn: sqlite3.Connection):
    """
    Load `hair_pulling_behaviours_patterns`, map awareness,
    tag relapse_risk, then overwrite the original table.
    """
    df = pd.read_sql_query("SELECT * FROM hair_pulling_behaviours_patterns", conn)

    # awareness → numeric
    if 'pulling_awareness' in df.columns:
        awareness_map = {'Yes':1.0, 'Sometimes':0.5, 'No':0.0}
        df['awareness_level_encoded'] = df['pulling_awareness'].map(awareness_map).fillna(0.0)
    else:
        df['awareness_level_encoded'] = 0.0

    # relapse risk tag
    def tag_risk(row):
        sev = row.get('pulling_severity', 0)
        aw  = row.get('awareness_level_encoded', 0)
        if sev >= 7 and aw <= 0.5:
            return 'high'
        elif sev >= 5:
            return 'moderate'
        else:
            return 'low'
    df['relapse_risk_tag'] = df.apply(tag_risk, axis=1)

    df.to_sql('hair_pulling_behaviours_patterns', conn, if_exists='replace', index=False)
    print(f"✅ Updated 'hair_pulling_behaviours_patterns' with awareness & relapse_risk ({len(df)} rows)")


# ── 3. BINARY YES/NO TABLE COUNTS ────────────────────────────────────────────────
def process_binary_tables(conn: sqlite3.Connection):
    """
    For every table named *_1_yes_0_no, sum the 1's per row, add it as
    `count_ones` column, and overwrite the original table.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
         WHERE type='table' AND name LIKE '%\_1\_yes\_0\_no' ESCAPE '\\'
    """)
    tables = [r[0] for r in cur.fetchall()]

    for tbl in tables:
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
        if 'id' not in df.columns:
            print(f"⚠️ Skipping '{tbl}' (no id column)")
            continue

        # identify binary columns (exclude id)
        bin_cols = [c for c in df.columns if c != 'id']
        # coerce to int and count 1's
        df[bin_cols] = df[bin_cols].apply(pd.to_numeric, errors='coerce') \
                                   .fillna(0).astype(int)
        df['count_ones'] = df[bin_cols].sum(axis=1)

        df.to_sql(tbl, conn, if_exists='replace', index=False)
        print(f"✅ Updated '{tbl}' with count_ones ({len(df)} rows)")


# ── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    """Function to run all preprocessing steps."""
    print("🔒 Starting preprocessing...")
    conn = sqlite3.connect(DB_PATH)
    try:
        print("🔄 Preprocessing demographics…")
        process_demographics(conn)
        print("🔄 Preprocessing behaviour patterns…")
        process_behaviour_patterns(conn)
        print("🔄 Preprocessing binary YES/NO tables…")
        process_binary_tables(conn)
    finally:
        conn.commit()
        conn.close()
        print("🔒 All preprocessing complete.")

if __name__ == "__main__":
    main()
