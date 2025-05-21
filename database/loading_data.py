#!/usr/bin/env python3
"""
preprocessing_data.py

Load every CSV-derived table from the TTM SQLite database,
outer-merge them into one coherent DataFrame keyed on `id`,
do light type conversions, and write out as
`survey_responses_merged` for EDA and modeling.

Usage:
    python preprocessing_data.py
"""
# === BEGINNING OF FILE ===
import sqlite3
import pandas as pd

# === CONFIG ===
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
OUTPUT_TABLE = "trichotillomania_data"

def get_table_names(conn):
    """
    Return all user tables in the SQLite database.
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return [row[0] for row in cursor.fetchall()]

def load_table(conn, name):
    """
    Load a single table into a DataFrame.
    If no 'id' column exists, create one from the row index.
    """
    df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    if 'id' not in df.columns:
        # fallback to row index if there's no explicit primary key
        df.insert(0, 'id', df.index.astype(int))
    return df

def merge_all_tables(conn, table_names):
    """
    Outer-merge all tables on 'id' so that no rows get dropped.
    """
    dfs = {}
    for tbl in table_names:
        dfs[tbl] = load_table(conn, tbl)

    # start from the first table
    merged = None
    for tbl, df in dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='id', how='outer', validate="one_to_one")
    return merged

def clean_types(df):
    """
    Parse timestamps, cast any 1/0 flags to int, and reset the index.
    """
    # parse any timestamp-like columns
    for col in df.columns:
        if 'timestamp' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # convert any boolean-ish or 1/0 columns to int
    for col in df.select_dtypes(include=['object']):
        # if all values are '0' or '1', cast
        unique = df[col].dropna().unique()
        if set(unique) <= {'0', '1'}:
            df[col] = df[col].astype(int)

    df.reset_index(drop=True, inplace=True)
    return df

def main():
    # 1. Connect
    conn = sqlite3.connect(DB_PATH)

    # 2. Introspect table names
    tables = get_table_names(conn)
    print("Found tables:", tables)

    # 3. Merge
    print("Merging tables on 'id'...", end=' ')
    merged = merge_all_tables(conn, tables)
    print("done. Shape:", merged.shape)

    # 4. Clean up types
    merged = clean_types(merged)

    # 5. Write back to DB
    print(f"Writing merged DataFrame to table '{OUTPUT_TABLE}'...", end=' ')
    merged.to_sql(OUTPUT_TABLE, conn, if_exists='replace', index=False)
    print("done.")

    conn.close()
    print("✅ preprocessing_data complete.")

if __name__ == "__main__":
    main()
# === END OF FILE ===
