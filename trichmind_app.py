"""
#!/usr/bin/env python
# trichmind_app.py

# ─── TRICHMIND STREAMLIT APP ──────────────────────────────────────────────────

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

Author: Girlie Razon
Date: 2025-05-26
"""

import os
import datetime
import sqlite3
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ─── 1. PAGE CONFIG ────────────────────────────────────────────────────────────
# logo should be in the same folder as this script:
LOGO_FILENAME = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\assets\logo.png"
if not os.path.exists(LOGO_FILENAME):
    st.warning(f"⚠️ Logo file not found at `{LOGO_FILENAME}`. Please place it next to this script.")

st.set_page_config(
    page_title="TrichMind",
    page_icon=LOGO_FILENAME if os.path.exists(LOGO_FILENAME) else "🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 2. CUSTOM CSS & PALETTE ───────────────────────────────────────────────────
BACKGROUND = "#E0F2F1"
CARD_BG    = "#FFFFFF"
PRIMARY    = "#00695C"
SECONDARY  = "#26A69A"
ACCENT     = "#80CBC4"
TEXT       = "#004D40"

st.markdown(f"""
<style>
/* App background */
.stApp {{ background-color: {BACKGROUND}; }}

/* Reduce padding */
.css-18e3th9 {{ padding: 1rem 2rem; }}

/* Center title area */
.main > div:nth-child(1) {{ display: flex; justify-content: center; }}

/* Metric cards */
.stMetric {{
  background-color: {CARD_BG} !important;
  border-radius: 12px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}}

/* DataFrame & tables styling */
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

/* Hide default footer */
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─── 3. LOGO DISPLAY ────────────────────────────────────────────────────────────
if os.path.exists(LOGO_FILENAME):
    st.image(LOGO_FILENAME, width=100)

st.title("TrichMind Dashboard")

# ─── 4. LOAD MODEL & DATA ──────────────────────────────────────────────────────
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

# ─── 5. APP TABS ───────────────────────────────────────────────────────────────
tab_home, tab_emotion, tab_journal, tab_triggers = st.tabs([
    "🏠 Home", "😊 Emotion Assistant", "📓 Journal & Progress", "📊 Triggers & Insights"
])

# ─── HOME TAB ───────────────────────────────────────────────────────────────────
with tab_home:
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.subheader("Relapse Risk")
        latest   = df.sort_values("timestamp").iloc[-1]
        feats    = model.feature_names_in_
        X_latest = pd.DataFrame([latest[feats]])
        pred     = label_enc.inverse_transform(model.predict(X_latest))[0]
        st.metric("Risk Level", pred.upper())
        st.caption(f"as of {latest.timestamp.date()}")

    with c2:
        st.subheader("Today's Entries")
        today = df[df.timestamp.dt.date == datetime.date.today()]
        st.metric("Entries recorded", len(today))

    with c3:
        st.subheader("Popular Coping Strategies")
        top = (
            df["effective_coping_strategies"].dropna()
              .str.split(",", expand=True)
              .stack().str.strip()
              .value_counts().head(5)
        )
        st.table(top.rename_axis("Strategy").reset_index(name="Count"))

# ─── EMOTION ASSISTANT TAB ─────────────────────────────────────────────────────
with tab_emotion:
    st.subheader("Emotion Assistant")
    feeling = st.text_input("How are you feeling?", placeholder="I'm stressed about...")
    if st.button("Get Suggestions"):
        txt = feeling.lower()
        if any(k in txt for k in ["stress","anx"]):
            sugg = ["Deep breathing", "Short walk", "Guided meditation"]
        elif "sad" in txt:
            sugg = ["Listen to uplifting music", "Write in journal"]
        else:
            sugg = ["Take a break", "Call a friend"]
        st.markdown("**Suggestions:**")
        for s in sugg:
            st.markdown(f"- {s}")

# ─── JOURNAL & PROGRESS TAB ────────────────────────────────────────────────────
with tab_journal:
    st.subheader("Journal & Progress")
    sel_date = st.date_input("Select date", value=datetime.date.today())
    subset  = df[df.timestamp.dt.date == sel_date]
    st.markdown(f"**Entries on {sel_date}:** {len(subset)}")
    weekly  = df.set_index("timestamp")["emotional_trigger_score"].resample("W").mean()
    st.line_chart(weekly, use_container_width=True)
    if st.checkbox("Show raw logs"):
        st.dataframe(subset[[
            "timestamp","pulling_severity","emotional_trigger_score",
            "coping_strategies","pulling_triggers"
        ]])

# ─── TRIGGERS & INSIGHTS TAB ──────────────────────────────────────────────────
with tab_triggers:
    st.subheader("Top 10 Emotional Triggers")
    counts = (
        df["pulling_triggers"].dropna()
          .str.split(",", expand=True).stack()
          .str.strip().value_counts().head(10)
    )
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        labels={"x":"Count","y":"Trigger"}, title="Top Triggers"
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
    mc = df["month"].value_counts().sort_index()
    fig3 = px.line(
        x=mc.index, y=mc.values, markers=True,
        labels={"x":"Month","y":"Entries"},
        title="Entries by Month"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='opacity:0.3;'/>"
    "<p style='text-align:center; color:#555;'>© 2025 TrichMind Research • Data remains anonymous</p>",
    unsafe_allow_html=True
)
