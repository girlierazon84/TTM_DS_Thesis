#!/usr/bin/env python3
"""
eda_analysis.py

This script performs Exploratory Data Analysis (EDA) on your TTM SQLite database,
which contains:

  * demographic
  * hair_pulling_behaviours_patterns
  * any number of *_1_yes_0_no binary tables

It generates and saves static visualizations (PNG) for:

- **Demographics**: age, gender, country, years_since_onset
- **TTM History & Behavior**: age_of_onset, years_since_onset, family_history,
  pulling_frequency_encoded, common_pulling_time, pulling_environment, relapse_risk_tag
- **Binary flags**: for each *_1_yes_0_no table, per-column “Yes” counts
  (automatically strips out the “_1_yes_0_no” suffix)

All outputs land in `figures/png/`.
"""
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "database", "ttm_database.db")
FIG_DIR  = os.path.join(BASE_DIR, "figures", "png")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_fig(fig, fname):
    """Tight-layout + save PNG."""
    path = os.path.join(FIG_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {fname}")


def plot_hist(df, col, title, fname, bins=20):
    """Numeric histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=bins, kde=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    save_fig(fig, fname)


def plot_count(df, col, title, fname, top_n=None):
    """Categorical countplot."""
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


def plot_box(df, by, col, title, fname):
    """Boxplot of col by by."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=by, y=col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(by.replace("_", " ").title())
    ax.set_ylabel(col.replace("_", " ").title())
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, fname)


def load_demographics(conn):
    """Load demographic table and compute years_since_onset if possible."""
    df = pd.read_sql_query("SELECT * FROM demographics", conn, parse_dates=["timestamp"])
    if {"age", "age_of_onset"}.issubset(df.columns):
        df["years_since_onset"] = df["age"] - df["age_of_onset"]
    return df


def load_behaviour(conn):
    """Load hair_pulling_behaviours_patterns and compute features + relapse risk."""
    df = pd.read_sql_query("SELECT * FROM hair_pulling_behaviours_patterns", conn, parse_dates=["timestamp"])
    if "pulling_frequency" in df.columns:
        freq_map = {"Daily":5, "Several times a week":4, "Weekly":3, "Monthly":2, "Rarely":1}
        df["pulling_frequency_encoded"] = df["pulling_frequency"].map(freq_map).fillna(0).astype(int)
    if "pulling_awareness" in df.columns:
        aw_map = {"Yes":1.0, "Sometimes":0.5, "No":0.0}
        df["awareness_level_encoded"] = df["pulling_awareness"].map(aw_map).fillna(0.0)
    def tag_risk(r):
        sev = r.get("pulling_severity", 0)
        aw  = r.get("awareness_level_encoded", 0)
        if sev >= 7 and aw <= 0.5: return "high"
        if sev >= 5: return "moderate"
        return "low"
    df["relapse_risk_tag"] = df.apply(tag_risk, axis=1)
    return df


def find_binary_tables(conn):
    """Find all tables with the pattern *_1_yes_0_no."""
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
         WHERE type='table' AND name GLOB '*_1_yes_0_no'
    """)
    return [row[0] for row in cur.fetchall()]


def plot_binary_table(conn, tbl):
    """
    For a given *_1_yes_0_no table, sum only the original binary columns,
    prettify column & table names (strip '_1_yes_0_no'), and plot a horizontal
    bar chart of Yes-counts.
    """
    df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
    if "id" not in df.columns:
        print(f"⚠️ skipping {tbl} (no id)")
        return

    # Only original binary columns, exclude any 'count_ones'
    bin_cols = [c for c in df.columns if c not in ("id", "count_ones")]
    yes_counts = (
        df[bin_cols]
        .apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        .sum()
    )

    # Prettify table name
    pretty_tbl = re.sub(r"_1_yes_0_no$", "", tbl).replace("_", " ").title()

    # Prettify feature names by stripping trailing pattern
    features = yes_counts.index.to_series() \
        .apply(lambda x: re.sub(r"_1_yes_0_no$", "", x).replace("_", " ").title())
    counts = yes_counts.values

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts, y=features, orient="h", ax=ax,
                color=sns.color_palette("muted")[2])
    ax.set_title(f"{pretty_tbl}: Yes Counts per Feature")
    ax.set_xlabel("Number of Yes Responses")
    ax.set_ylabel("Feature")
    save_fig(fig, f"{tbl}_yes_counts.png")


def main():
    """Main function to run EDA."""
    print("🔍 Starting EDA...")
    conn = sqlite3.connect(DB_PATH)

    # 1) Demographics EDA
    demo = load_demographics(conn)
    if "age" in demo:
        plot_hist(demo, "age", "Age Distribution", "age.png")
    if "gender" in demo:
        plot_count(demo, "gender", "Gender Breakdown", "gender.png")
    if "country" in demo:
        plot_count(demo, "country", "Country (Top 10)", "country.png", top_n=10)
    if "years_since_onset" in demo:
        plot_hist(demo, "years_since_onset", "Years Since Onset", "years_since_onset.png")
    if "family_history" in demo:
        plot_count(demo, "family_history", "Family History", "family_history.png")

    # 2) Behaviour Patterns EDA
    beh = load_behaviour(conn)
    if "age_of_onset" in beh:
        plot_hist(beh, "age_of_onset", "Age of Onset", "age_of_onset.png")
    if "pulling_frequency_encoded" in beh:
        plot_hist(beh, "pulling_frequency_encoded", "Pulling Frequency (Encoded)", "freq_encoded.png")
    if "common_pulling_time" in beh:
        plot_count(beh, "common_pulling_time", "Common Pulling Time", "common_time.png")
    if "pulling_environment" in beh:
        plot_count(beh, "pulling_environment", "Pulling Environment", "environment.png")
    if "pulling_severity" in beh:
        plot_hist(beh, "pulling_severity", "Pulling Severity (1–10)", "severity.png")
    if "relapse_risk_tag" in beh:
        plot_count(beh, "relapse_risk_tag", "Relapse Risk Category", "relapse_risk.png")

    # 3) Binary-flag tables EDA
    for tbl in find_binary_tables(conn):
        plot_binary_table(conn, tbl)

    conn.close()
    print("✅ EDA complete — all figures are in figures/png/")


if __name__ == "__main__":
    main()
# ── END ───────────────────────────────────────────────────────────────────────
