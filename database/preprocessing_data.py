"""
preprocessing.py

This script loads raw survey responses from a semicolon-separated CSV,
performs data cleaning and feature engineering (including encoding categorical
variables and tagging relapse risk), and writes the cleaned data directly into
a new SQLite database table for downstream modeling and analysis.

Functions:
- load_data(csv_path: str) -> pd.DataFrame
    Load the raw survey CSV into a DataFrame.

- preprocess_data(df: pd.DataFrame) -> pd.DataFrame
    Clean the DataFrame, convert types, handle missing values, engineer
    new features, encode categorical variables, and tag relapse risk.

- save_preprocessed_to_db(df: pd.DataFrame, db_path: str, table_name: str) -> None
    Save the preprocessed DataFrame into a SQLite table (replacing if exists).

Usage:
    python preprocessing.py
"""

import sqlite3
import pandas as pd

# === Constants ===
RAW_CSV      = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\survey_responses.csv"
DB_PATH      = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
TABLE_NAME   = "trichotillomania_data"

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw survey data from a semicolon-separated CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw survey responses.
    """
    return pd.read_csv(csv_path, sep=';', encoding='utf-8', on_bad_lines='skip')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features for modeling.

    Steps
    -----
    1. Parse `timestamp` to datetime.
    2. Drop rows missing key fields.
    3. Compute `years_since_onset`.
    4. Encode categorical variables.
    5. Compute text-derived counts.
    6. One-hot encode `gender`.
    7. Tag relapse risk.
    8. Reset index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw survey DataFrame.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for database storage and modeling.
    """
    df = df.copy()

    # 1. Timestamp → datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # 2. Drop rows missing essentials
    essentials = ['age','age_of_onset','pulling_severity','pulling_frequency','pulling_awareness']
    df.dropna(subset=essentials, inplace=True)

    # 3. Years since onset
    df['years_since_onset'] = df['age'] - df['age_of_onset']

    # 4. Encode categorical variables
    freq_map = {'Daily':5,'Several times a week':4,'Weekly':3,'Monthly':2,'Rarely':1}
    df['pulling_frequency_encoded'] = df['pulling_frequency'].map(freq_map).fillna(0).astype(int)

    awareness_map = {'Yes':1.0,'Sometimes':0.5,'No':0.0}
    df['awareness_level_encoded'] = df['pulling_awareness'].map(awareness_map).fillna(0.0)

    df['seasonal_change_binary'] = df['seasonal_change'].map({'Yes':1,'No':0}).fillna(0).astype(int)
    df['support_sought_binary']  = df['support_sought'].map({'Yes':1,'No':0}).fillna(0).astype(int)
    df['therapy_sought_binary']  = df['therapy_sought'].map({'Yes':1,'No':0}).fillna(0).astype(int)
    df['stopped_binary']         = df['successfully_stopped'].map({'Yes':1,'No':0}).fillna(0).astype(int)

    # 5. Compute text counts
    df['emotional_trigger_score']       = df['emotions_before_pulling'].fillna('').apply(lambda x: len(x.split(',')))
    df['coping_strategies_count']       = df['coping_strategies'].fillna('').apply(lambda x: len(x.split(',')))
    df['mental_health_condition_count'] = df['other_mental_conditions'].fillna('').apply(lambda x: len(x.split(',')))
    df['activities_count']              = df['activities_lists'].fillna('').apply(lambda x: len(x.split(',')))
    df['seasons_affected_count']        = df['seasons_affected'].fillna('').apply(lambda x: len(x.split(',')))

    # 6. One-hot encode gender
    df = pd.get_dummies(df, columns=['gender'], prefix='gender')

    # 7. Tag relapse risk
    def tag_risk(row):
        if row['pulling_severity'] >= 7 and row['awareness_level_encoded'] <= 0.5:
            return 'high'
        elif row['pulling_severity'] >= 5:
            return 'moderate'
        else:
            return 'low'
    df['relapse_risk_tag'] = df.apply(tag_risk, axis=1)

    # 8. Reset index
    df.reset_index(drop=True, inplace=True)
    return df

def save_preprocessed_to_db(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    Save the preprocessed DataFrame into a SQLite database table.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    db_path : str
        Path to the SQLite .db file.
    table_name : str
        Name of the table to create/replace.
    """
    conn = sqlite3.connect(db_path)
    # Replace existing table
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅ Table '{table_name}' written to {db_path}")

if __name__ == "__main__":
    # 1. Load raw CSV
    df_raw = load_data(RAW_CSV)

    # 2. Preprocess
    df_clean = preprocess_data(df_raw)

    # 3. Write cleaned data to new SQLite database
    save_preprocessed_to_db(df_clean, DB_PATH, TABLE_NAME)

    print("✅ Preprocessing complete; cleaned data saved to database.")
# End of script
