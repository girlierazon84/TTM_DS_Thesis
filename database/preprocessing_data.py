"""
preprocessing.py

This script loads survey responses from a SQLite database, performs data
cleaning and feature engineering (including encoding categorical variables
and tagging relapse risk), and saves a preprocessed CSV for modeling and analysis.

Functions:
- load_data(db_path: str) -> pd.DataFrame
    Load the `survey_responses` table from a SQLite database into a DataFrame.

- preprocess_data(df: pd.DataFrame) -> pd.DataFrame
    Clean the DataFrame, convert types, handle missing values, engineer
    new features, encode categorical variables, and tag relapse risk.

- save_preprocessed_data(df: pd.DataFrame, output_path: str) -> None
    Save the preprocessed DataFrame to a CSV file.

Usage:
    from preprocessing import load_data, preprocess_data, save_preprocessed_data

    df = load_data(DB_PATH)
    df_clean = preprocess_data(df)
    save_preprocessed_data(df_clean, OUTPUT_CSV)
"""

import os
import sqlite3
import pandas as pd

# === Constants ===
DB_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\ttm_database.db"
OUTPUT_CSV = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\database\preprocessed_ttm_data.csv"

def load_data(db_path: str) -> pd.DataFrame:
    """
    Load the trichotillomania_data table from a SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite .db file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all rows from trichotillomania_data.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM trichotillomania_data", conn)
    conn.close()
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features for modeling.

    Steps
    -----
    1. Convert `timestamp` to datetime.
    2. Drop rows with missing essential fields.
    3. Compute `years_since_onset`.
    4. Encode categorical variables:
       - pulling_frequency → pulling_frequency_encoded
       - pulling_awareness → awareness_level_encoded
       - seasonal_change → seasonal_change_binary
       - support_sought → support_sought_binary
       - therapy_sought → therapy_sought_binary
       - successfully_stopped → stopped_binary
    5. Compute counts:
       - emotional_trigger_score
       - coping_strategies_count
       - mental_health_condition_count
       - activities_count
       - seasons_affected_count
    6. One-hot encode `gender`.
    7. Tag relapse risk.
    8. Reset index and return.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from the database.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for modeling.
    """
    df = df.copy()

    # 1. Timestamp → datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # 2. Drop rows missing essential fields
    essential_cols = ['age', 'age_of_onset', 'pulling_severity',
                      'pulling_frequency', 'pulling_awareness']
    df.dropna(subset=essential_cols, inplace=True)

    # 3. Years since onset
    df['years_since_onset'] = df['age'] - df['age_of_onset']

    # 4. Encode categorical variables
    freq_map = {'Daily': 5, 'Several times a week': 4, 'Weekly': 3, 'Monthly': 2, 'Rarely': 1}
    df['pulling_frequency_encoded'] = df['pulling_frequency'].map(freq_map).fillna(0).astype(int)

    awareness_map = {'Yes': 1.0, 'Sometimes': 0.5, 'No': 0.0}
    df['awareness_level_encoded'] = df['pulling_awareness'].map(awareness_map).fillna(0.0)

    df['seasonal_change_binary']  = df['seasonal_change'].map({'Yes':1, 'No':0}).fillna(0).astype(int)
    df['support_sought_binary']   = df['support_sought'].map({'Yes':1, 'No':0}).fillna(0).astype(int)
    df['therapy_sought_binary']   = df['therapy_sought'].map({'Yes':1, 'No':0}).fillna(0).astype(int)
    df['stopped_binary']          = df['successfully_stopped'].map({'Yes':1, 'No':0}).fillna(0).astype(int)

    # 5. Compute textual counts
    df['emotional_trigger_score']       = df['emotions_before_pulling'].fillna('').apply(lambda x: len(x.split(',')))
    df['coping_strategies_count']       = df['coping_strategies'].fillna('').apply(lambda x: len(x.split(',')))
    df['mental_health_condition_count'] = df['other_mental_conditions'].fillna('').apply(lambda x: len(x.split(',')))
    df['activities_count']              = df['activities_lists'].fillna('').apply(lambda x: len(x.split(',')))
    df['seasons_affected_count']        = df['seasons_affected'].fillna('').apply(lambda x: len(x.split(',')))

    # 6. One-hot encode gender
    df = pd.get_dummies(df, columns=['gender'], prefix='gender')

    # 7. Tag relapse risk
    def tag_risk(row):
        """Tag the risk of relapse based on severity and awareness level."""
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

def save_preprocessed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the preprocessed DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed DataFrame.
    output_path : str
        Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    df_raw   = load_data(DB_PATH)
    df_clean = preprocess_data(df_raw)
    save_preprocessed_data(df_clean, OUTPUT_CSV)
    print("✅ Data preprocessing complete.")
# === End of preprocessing.py ===
