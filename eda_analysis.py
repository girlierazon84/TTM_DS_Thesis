#!/usr/bin/env python3
"""
eda_analysis.py

Perform Exploratory Data Analysis on your TTM SQLite database, which has:

  * demographics
  * hair_pulling_behaviours_patterns
  * any number of *_1_yes_0_no binary tables

Outputs static PNGs into `figures/png/`:

- Demographics: age, gender, country, years_since_onset, family_history
- Behavior: age_of_onset, pulling_frequency_encoded, pulling_environment,
  pulling_severity, relapse_risk_tag
- Binary‐flag tables: horizontal bars of total “Yes” counts per feature
- Correlation heatmap
"""
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "database", "ttm_database.db")
FIG_DIR  = os.path.join(BASE_DIR, "figures", "png")
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def save_fig(fig: plt.Figure, name: str):
    """Save a figure as a PNG into FIG_DIR."""
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✅ {name}")


def plot_hist(df: pd.DataFrame, col: str, title: str, fname: str, bins: int = 20):
    """Plot a histogram of a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=bins, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    save_fig(fig, fname)


def plot_count(df: pd.DataFrame, col: str, title: str, fname: str, top_n: int = None):
    """Plot a bar‐count plot of a categorical column."""
    data = df[col].dropna().astype(str)
    if top_n:
        top = data.value_counts().nlargest(top_n).index
        data = data[data.isin(top)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=data, order=data.value_counts().index, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, fname)


def plot_heatmap(df: pd.DataFrame, cols: list, title: str, fname: str):
    """Plot a correlation heatmap for given numeric columns."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, ax=ax)
    ax.set_title(title)
    save_fig(fig, fname)


def load_demographics(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load and augment the demographics table."""
    df = pd.read_sql("SELECT * FROM demographics", conn)
    if {"age", "age_of_onset"}.issubset(df.columns):
        df["years_since_onset"] = df["age"] - df["age_of_onset"]
    return df


def load_behaviour(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load and augment the hair_pulling_behaviours_patterns table."""
    df = pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns", conn)
    if "pulling_frequency" in df:
        freq_map = {"Daily":5, "Several times a week":4, "Weekly":3, "Monthly":2, "Rarely":1}
        df["pulling_frequency_encoded"] = df["pulling_frequency"].map(freq_map).fillna(0).astype(int)
    if "pulling_awareness" in df:
        aw_map = {"Yes":1.0, "Sometimes":0.5, "No":0.0}
        df["awareness_level_encoded"] = df["pulling_awareness"].map(aw_map).fillna(0.0)
    def tag_risk(r):
        sev = r.get("pulling_severity", 0)
        aw  = r.get("awareness_level_encoded", 0)
        if sev >= 7 and aw <= 0.5:
            return "high"
        elif sev >= 5:
            return "moderate"
        else:
            return "low"
    df["relapse_risk_tag"] = df.apply(tag_risk, axis=1)
    return df


def plot_binary(conn: sqlite3.Connection, tbl: str):
    """
    For a binary table ending in _1_yes_0_no, plot
    total 'Yes' counts per original feature column.
    """
    df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
    if "id" not in df.columns:
        return
    bin_cols = [c for c in df.columns if c != "id"]
    yes_counts = df[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).sum()
    labels = [c.replace("_1_yes_0_no", "").replace("_", " ").title() for c in yes_counts.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=yes_counts.values, y=labels, ax=ax, color="steelblue")
    ax.set_title(tbl.replace("_", " ").title())
    ax.set_xlabel("Yes Count")
    ax.set_ylabel("Feature")
    save_fig(fig, f"{tbl}_yes_counts.png")


def main():
    """Run the full EDA suite."""
    conn = sqlite3.connect(DB_PATH)

    # --- Demographics ---
    demo = load_demographics(conn)
    plot_hist(demo, "age", "Age Distribution", "demographics_age.png")
    if "gender" in demo:
        plot_count(demo, "gender", "Gender Breakdown", "demographics_gender.png")
    if "country" in demo:
        plot_count(demo, "country", "Country of Residence", "demographics_country.png", top_n=10)
    if "years_since_onset" in demo:
        plot_hist(demo, "years_since_onset", "Years Since Onset", "demographics_years_since_onset.png")
    if "family_history" in demo:
        plot_count(demo, "family_history", "Family History (1=Yes,0=No)", "demographics_family_history.png")

    # --- Behaviour Patterns ---
    beh = load_behaviour(conn)
    if "age_of_onset" in beh:
        plot_hist(beh, "age_of_onset", "Age of Onset", "behaviour_age_of_onset.png")
    if "pulling_frequency_encoded" in beh:
        plot_hist(beh, "pulling_frequency_encoded", "Pulling Frequency (Encoded)", "behaviour_freq_encoded.png")
    if "common_pulling_time" in beh:
        plot_count(beh, "common_pulling_time", "Common Pulling Time", "behaviour_common_time.png")
    if "pulling_environment" in beh:
        plot_count(beh, "pulling_environment", "Pulling Environment", "behaviour_environment.png")
    if "pulling_severity" in beh:
        plot_hist(beh, "pulling_severity", "Pulling Severity (1–10)", "behaviour_severity.png")
    if "relapse_risk_tag" in beh:
        plot_count(beh, "relapse_risk_tag", "Relapse Risk Category", "behaviour_relapse_risk.png")

    # --- Binary flag tables ---
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_1_yes_0_no'")
    for (tbl,) in cur.fetchall():
        plot_binary(conn, tbl)

    # --- Correlation Heatmap (on behaviour numeric cols) ---
    num_cols = [c for c in beh.select_dtypes("number").columns if c != "id"]
    if len(num_cols) > 1:
        plot_heatmap(beh, num_cols, "Behaviour: Correlation Heatmap", "behaviour_corr_heatmap.png")

    conn.close()


if __name__ == "__main__":
    main()
