#!/usr/bin/env python
# trichmind_app.py

"""
TrichMind Streamlit App

A Streamlit application designed to help individuals manage
trichotillomania (hair-pulling disorder) by providing insights into
their behavior, emotional triggers, and coping strategies. The app
uses a trained model to predict relapse risk based on user input and history.

Features:
- Relapse Risk Prediction: Real-time risk level.
- Home: Daily summary, popular strategies.
- Emotion Assistant: On-the-fly coping suggestions.
- Journal & Progress: Time-series and raw logs.
- Triggers & Insights: Top triggers, environments, seasonal patterns.

Author: Your Name
Date: 2025-05-XX
"""

# ─── 1. IMPORTS & PAGE CONFIG ──────────────────────────────────────────────────
import datetime
import os
import sqlite3
import joblib

import pandas as pd
import plotly.express as px
import streamlit as st

# Must be the first Streamlit call
st.set_page_config(
    page_title="TrichMind 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 2. NEW COLOR PALETTE & CUSTOM CSS ─────────────────────────────────────────
# Mint-teal theme
BACKGROUND = "#E0F2F1"
CARD_BG     = "#FFFFFF"
PRIMARY     = "#00695C"
SECONDARY   = "#26A69A"
ACCENT      = "#80CBC4"
TEXT        = "#004D40"

st.markdown(f"""
    <style>

    /* Page background */
    .stApp {{ background-color: {BACKGROUND}; }}

    /* Reduce main padding */
    .css-18e3th9 {{ padding: 1rem 2rem; }}

    /* Center title */
    .main > div:nth-child(1) {{ display: flex; justify-content: center; }}

    /* Card containers */
    .stMetric {{
        background-color: {CARD_BG} !important;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1rem;
    }}

    .stDataFrame, .stTable {{
        background-color: {CARD_BG} !important;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {SECONDARY} !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }}
    .stButton > button:hover {{
        background-color: {ACCENT} !important;
    }}

    /* Tabs */
    .stTabs [role="tab"] {{
        background-color: {CARD_BG};
        color: {TEXT};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-right: 0.3rem;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background-color: {PRIMARY};
        color: white !important;
    }}

    /* Subheaders */
    .css-1v0mbdj h2 {{
        color: {PRIMARY} !important;
    }}

    /* Footer */
    footer {{ visibility: hidden; }}
    </style>
""", unsafe_allow_html=True)

# ─── 3. LOAD MODEL & DATA ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    """Load the trained pipeline and label encoder."""
    model = joblib.load("models/best_model.pkl")
    le    = joblib.load("models/label_encoder.pkl")
    return model, le

@st.cache_data(ttl=600)
def load_data():
    """Load survey responses from SQLite into a DataFrame."""
    conn = sqlite3.connect("database/ttm_database.db", check_same_thread=False)
    df   = pd.read_sql_query(
        "SELECT * FROM trichotillomania_data", conn,
        parse_dates=["timestamp"]
    )
    conn.close()
    return df

model, label_enc = load_model_and_encoder()
df = load_data()

# ─── 4. APP LAYOUT ─────────────────────────────────────────────────────────────
st.title("🧠 TrichMind Dashboard")

tab_home, tab_emotion, tab_journal, tab_triggers = st.tabs([
    "🏠 Home", "😊 Emotion Assistant", "📓 Journal & Progress", "📊 Triggers & Insights"
])

# ─── HOME TAB ───────────────────────────────────────────────────────────────────
with tab_home:
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.subheader("Relapse Risk")
        latest = df.sort_values("timestamp").iloc[-1]
        feat_cols = list(model.feature_names_in_)
        X_latest = pd.DataFrame([latest[feat_cols]])
        pred = label_enc.inverse_transform(model.predict(X_latest))[0]
        st.metric(label="Risk Level", value=pred.upper())
        st.caption(f"as of {latest.timestamp.date()}")

    with c2:
        st.subheader("Today's Entries")
        today = df[df.timestamp.dt.date == datetime.date.today()]
        st.metric(label="Entries recorded", value=len(today))

    with c3:
        st.subheader("Popular Coping Strategies")
        top_coping = (
            df["effective_coping_strategies"]
            .dropna().str.split(",",expand=True)
            .stack().str.strip()
            .value_counts().head(5)
        )
        st.table(top_coping.rename_axis("Strategy").reset_index(name="Count"))

# ─── EMOTION ASSISTANT TAB ─────────────────────────────────────────────────────
with tab_emotion:
    st.subheader("Emotion Assistant")
    txt = st.text_input("How are you feeling?", placeholder="I'm stressed about...")
    if st.button("Get Suggestions"):
        msg = txt.lower()
        if "stress" in msg or "anx" in msg:
            sugg = ["Deep breathing", "Short walk", "Guided meditation"]
        elif "sad" in msg:
            sugg = ["Listen to uplifting music", "Write in journal"]
        else:
            sugg = ["Take a break", "Call a friend"]
        for s in sugg:
            st.markdown(f"- {s}")

# ─── JOURNAL & PROGRESS TAB ────────────────────────────────────────────────────
with tab_journal:
    st.subheader("Journal & Progress")
    d = st.date_input("Select date", value=datetime.date.today())
    day = df[df.timestamp.dt.date == d]
    st.markdown(f"**Entries on {d}:** {len(day)}")
    weekly = df.set_index("timestamp")["emotional_trigger_score"].resample("W").mean()
    st.line_chart(weekly, use_container_width=True)
    if st.checkbox("Show raw logs"):
        st.dataframe(day[[
            "timestamp", "pulling_severity", "emotional_trigger_score",
            "coping_strategies", "pulling_triggers"
        ]])

# ─── TRIGGERS & INSIGHTS TAB ──────────────────────────────────────────────────
with tab_triggers:
    st.subheader("Top 10 Emotional Triggers")
    tcounts = (
        df["pulling_triggers"].dropna()
        .str.split(",", expand=True).stack()
        .str.strip().value_counts().head(10)
    )
    fig = px.bar(
        x=tcounts.values, y=tcounts.index, orientation="h",
        labels={"x":"Count","y":"Trigger"},
        title="Top Triggers"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pulling Environment")
    env = df["pulling_environment"].value_counts()
    fig2 = px.pie(
        names=env.index, values=env.values,
        title="Where are you pulling?"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Seasonal Patterns")
    df["month"] = df.timestamp.dt.month
    mcounts = df["month"].value_counts().sort_index()
    fig3 = px.line(
        x=mcounts.index, y=mcounts.values, markers=True,
        labels={"x":"Month","y":"Entries"},
        title="Entries by Month"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='opacity:0.3;'><p style='text-align:center; color:#555;'>"
    "© 2025 TrichMind Research • Data remains anonymous</p>",
    unsafe_allow_html=True
)
# ─── END OF SCRIPT ─────────────────────────────────────────────────────────────
