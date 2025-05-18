"""
eda_ttmsurvey.py

This script performs Exploratory Data Analysis (EDA) on processed Trichotillomania
survey data stored in a SQLite database. It generates and saves rich static
(Matplotlib) visualizations for key variables and relationships relevant to the
thesis and app development.

All plots are saved as PNG files in the `figures/png/` directory.

Key Analyses:
- Demographics distributions (age, gender, country)
- TTM history (age_of_onset, years_since_onset, family_history)
- Behavior patterns (pulling_frequency_encoded, common_pulling_time, pulling_environment)
- Emotional triggers (emotional_trigger_score, pulling_triggers)
- Coping & control (coping_strategies_count, support_sought)
- Severity & relapse (pulling_severity, relapse_risk_tag, successfully_stopped)
- Seasonal patterns and time trends (timestamp)
- Additional insights: boxplots, scatterplots with trend lines, pie charts,
  correlation heatmap
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Setup paths ---
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'database', 'ttm_database.db')
TABLE_NAME = 'trichotillomania_data'
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'png')
os.makedirs(FIG_DIR, exist_ok=True)

def load_data(db_path: str, table: str) -> pd.DataFrame:
    """
    Load the cleaned survey data from SQLite and parse timestamp.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    table : str
        Name of the table containing cleaned responses.

    Returns
    -------
    pd.DataFrame
        DataFrame with all survey responses.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def save_fig(fig, fname):
    """
    Save a Matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    fname : str
        Filename (PNG) under FIG_DIR.
    """
    path = os.path.join(FIG_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {fname}")

def hist_with_box(df, col, title, fname, bins=20):
    """
    Plot a histogram with an inset boxplot.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    title : str
    fname : str
    bins : int
    """
    fig, ax = plt.subplots(figsize=(8,5))
    data = df[col].dropna()
    ax.hist(data, bins=bins, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(col.replace('_',' ').title())
    ax.set_ylabel('Count')

    # inset boxplot
    ax_box = inset_axes(ax, width="30%", height="15%", loc='upper right')
    ax_box.boxplot(data, vert=False)
    ax_box.set_yticks([])
    ax_box.set_xlabel('')

    save_fig(fig, fname)

def bar_with_counts(df, col, title, fname, top_n=None, horizontal=False):
    """
    Plot bar chart with counts, optionally top_n categories.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    title : str
    fname : str
    top_n : int or None
    horizontal : bool
    """
    counts = df[col].dropna().value_counts()
    if top_n:
        counts = counts.head(top_n)
    fig, ax = plt.subplots(figsize=(8,5))
    if horizontal:
        counts.sort_values().plot.barh(ax=ax)
        ax.set_ylabel(col.replace('_',' ').title())
        ax.set_xlabel('Count')
    else:
        counts.plot.bar(ax=ax)
        ax.set_xlabel(col.replace('_',' ').title())
        ax.set_ylabel('Count')
    ax.set_title(title)
    ax.xaxis.set_tick_params(rotation=45)
    save_fig(fig, fname)

def pie_chart(df, col, title, fname):
    """
    Plot a pie chart of value counts.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    title : str
    fname : str
    """
    counts = df[col].dropna().value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(title)
    save_fig(fig, fname)

def box_scatter_trend(df, x, y, title, fname):
    """
    Scatter plot x vs y with linear trend line plus a boxplot on margins.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
    title : str
    fname : str
    """
    clean = df[[x,y]].dropna()
    fig = plt.figure(figsize=(8,6))
    # main scatter
    ax = fig.add_axes([0.1,0.1,0.65,0.65])
    ax.scatter(clean[x], clean[y], alpha=0.6)
    ax.set_xlabel(x.replace('_',' ').title())
    ax.set_ylabel(y.replace('_',' ').title())
    ax.set_title(title)

    # fit line
    m, b = np.polyfit(clean[x], clean[y], 1)
    x_vals = np.array(ax.get_xlim())
    ax.plot(x_vals, m*x_vals + b, '--', color='red')

    # marginal histograms
    ax_xhist = inset_axes(ax, width="30%", height="15%", loc='upper right', borderpad=1)
    ax_xhist.hist(clean[y], bins=20, orientation='horizontal', edgecolor='black')
    ax_xhist.set_xticks([]); ax_xhist.set_yticks([])

    ax_yhist = inset_axes(ax, width="30%", height="15%", loc='lower left', borderpad=1)
    ax_yhist.hist(clean[x], bins=20, edgecolor='black')
    ax_yhist.set_xticks([]); ax_yhist.set_yticks([])

    save_fig(fig, fname)

def corr_heatmap(df, cols, title, fname):
    """
    Plot a correlation heatmap of numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
    title : str
    fname : str
    """
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c.replace('_',' ').title() for c in cols], rotation=45, ha='left')
    ax.set_yticklabels([c.replace('_',' ').title() for c in cols])
    ax.set_title(title, pad=20)
    save_fig(fig, fname)

def time_series(df, date_col, title, fname, freq='D'):
    """
    Plot a time series of response counts.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    title : str
    fname : str
    freq : str
    """
    ts = df.set_index(date_col).resample(freq).size()
    fig, ax = plt.subplots(figsize=(10,4))
    ts.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save_fig(fig, fname)

def main():
    """
    Main function to run EDA on the TTM survey data.
    """
    df = load_data(DB_PATH, TABLE_NAME)

    # Demographics
    if 'age' in df:
        hist_with_box(df, 'age', 'Age Distribution', 'age_distribution.png')
    if 'gender' in df:
        bar_with_counts(df, 'gender', 'Gender Breakdown', 'gender_breakdown.png')
        pie_chart(df, 'gender', 'Gender Share', 'gender_share.png')
    if 'country' in df:
        bar_with_counts(df, 'country', 'Responses by Country', 'country_counts.png', top_n=10)

    # TTM History
    for col,name in [('age_of_onset','Age of Onset'),('years_since_onset','Years Since Onset')]:
        if col in df:
            hist_with_box(df, col, name, f'{col}.png')
    if 'family_history' in df:
        bar_with_counts(df, 'family_history', 'Family History of TTM', 'family_history.png')

    # Behavior Patterns
    if 'pulling_frequency_encoded' in df:
        hist_with_box(df, 'pulling_frequency_encoded', 'Pulling Frequency (Encoded)', 'freq_encoded.png')
    for col,name in [('common_pulling_time','Common Pulling Time'),('pulling_environment','Pulling Environment')]:
        if col in df:
            bar_with_counts(df, col, name, f'{col}.png')

    # Severity vs Frequency
    if {'pulling_frequency_encoded','pulling_severity'}.issubset(df.columns):
        box_scatter_trend(df, 'pulling_frequency_encoded','pulling_severity',
                          'Severity vs. Frequency', 'severity_vs_frequency.png')

    # Emotional Triggers
    if 'emotional_trigger_score' in df:
        hist_with_box(df, 'emotional_trigger_score',
                     'Emotional Trigger Score Distribution','trigger_score.png')
    if 'pulling_triggers' in df:
        # explode multi-valued triggers
        counts = (
            df['pulling_triggers'].dropna().str.split(',', expand=True)
              .stack().str.strip().value_counts()
        )
        top = counts.head(10)
        bar_with_counts(pd.DataFrame({ 'trigger':top.index, 'count':top.values }),
                        'trigger','Top 10 Pulling Triggers','top_triggers.png', horizontal=True)

    # Coping & Control
    if 'coping_strategies_count' in df:
        hist_with_box(df, 'coping_strategies_count','Coping Strategies Count','coping_count.png')
    if 'support_sought' in df:
        bar_with_counts(df, 'support_sought','Support Sought','support_sought.png')

    # Severity & Relapse
    if 'pulling_severity' in df:
        hist_with_box(df, 'pulling_severity','Pulling Severity','severity.png')
    if 'relapse_risk_tag' in df:
        bar_with_counts(df, 'relapse_risk_tag','Relapse Risk Category','relapse_risk.png')
    if 'successfully_stopped' in df:
        bar_with_counts(df, 'successfully_stopped','Successfully Stopped','stopped.png')

    # Correlation Heatmap
    numeric = ['age','age_of_onset','years_since_onset',
               'pulling_frequency_encoded','awareness_level_encoded',
               'emotional_trigger_score','coping_strategies_count','pulling_severity']
    present = [c for c in numeric if c in df.columns]
    if len(present)>=2:
        corr_heatmap(df, present,'Correlation Heatmap','correlation.png')

    # Seasonal & Time Trends
    if 'timestamp' in df and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['month'] = df['timestamp'].dt.month
        bar_with_counts(df,'month','Responses by Month','monthly_counts.png', horizontal=True)
        time_series(df,'timestamp','Daily Response Counts','daily_responses.png')

    print("✅ EDA complete: all figures in 'figures/png/'")

if __name__ == '__main__':
    main()
# End of script
