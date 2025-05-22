#!/usr/bin/env python3
# trichmind_app.py

"""
TrichMind Streamlit App

A Streamlit application designed to help individuals manage
trichotillomania (hair-pulling disorder) by providing insights into
their behavior, emotional triggers, and coping strategies. The app
uses a trained model to predict relapse risk based on user input.

Features:
- Relapse Risk Prediction: via sliders for severity & awareness.
- Emotion Assistant: quick coping suggestions.
- Coping Strategies: top strategies from your dataset.
- Triggers & Insights: most common triggers & environments.
- Binary Flags Summary: bar charts of Yes-counts per feature.

Author: Girlie Razon
Date: 2025-05-26
"""

import os
import sqlite3

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ─── 1. PAGE CONFIG & LOGO ─────────────────────────────────────────────────────
LOGO_PATH = r"C:\Users\girli\OneDrive\Desktop\TTM_DS_Thesis\assets\logo.png"
icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "🧠"

st.set_page_config(
    page_title="TrichMind",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=100)

st.title("🧠 TrichMind Dashboard")

# ─── 2. STYLING ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #E0F2F1; }
  .css-18e3th9 { padding: 1rem 2rem; }
  .stTabs [role="tab"][aria-selected="true"] { background-color: #00695C; color: white !important; }
  .stButton>button { background-color: #26A69A !important; color: white !important; border-radius:8px; }
  .stButton>button:hover { background-color: #80CBC4 !important; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── 3. LOAD MODEL & DATA ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    """Load trained pipeline + label encoder."""
    m = joblib.load("models/best_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return m, le

@st.cache_data(ttl=600)
def load_all_data():
    """
    Load:
      - demographics
      - hair_pulling_behaviours_patterns
      - all *_1_yes_0_no tables
    Merge on `id`. Drop any 'count_ones' columns to avoid dups.
    """
    conn = sqlite3.connect("database/ttm_database.db", check_same_thread=False)

    # demographics → add years_since_onset
    demo = pd.read_sql("SELECT * FROM demographics", conn)
    if {"age","age_of_onset"}.issubset(demo.columns):
        demo["years_since_onset"] = demo["age"] - demo["age_of_onset"]

    # behaviour patterns → encode + relapse_risk_tag
    beh = pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns", conn)
    if "pulling_frequency" in beh:
        freq_map = {"Daily":5,"Several times a week":4,"Weekly":3,"Monthly":2,"Rarely":1}
        beh["pulling_frequency_encoded"] = beh["pulling_frequency"].map(freq_map).fillna(0).astype(int)
    if "pulling_awareness" in beh:
        aw_map = {"Yes":1.0,"Sometimes":0.5,"No":0.0}
        beh["awareness_level_encoded"] = beh["pulling_awareness"].map(aw_map).fillna(0.0)
    def tag(r):
        s, a = r.get("pulling_severity",0), r.get("awareness_level_encoded",0)
        if s>=7 and a<=0.5: return "high"
        if s>=5: return "moderate"
        return "low"
    beh["relapse_risk_tag"] = beh.apply(tag, axis=1)

    # merge
    df = demo.merge(beh, on="id", how="outer", suffixes=("_demo","_beh"))

    # binary tables
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%\\_1\\_yes\\_0\\_no' ESCAPE '\\'")
    for (tbl,) in cur.fetchall():
        b = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        if "count_ones" in b.columns:
            b = b.drop(columns=["count_ones"])
        df = df.merge(b, on="id", how="left")

    conn.close()
    return df

model, label_enc = load_model_and_encoder()
df = load_all_data()

# ─── 4. APP TABS ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home", "😊 Emotion Assistant", "📊 Triggers & Insights", "✅ Binary Flags"
])

# ─── HOME TAB ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Predict Your Relapse Risk")
    sev = st.slider("Pulling Severity (1–10)", 1, 10, 5)
    aw  = st.select_slider("Awareness Level", options=["No","Sometimes","Yes"], value="Sometimes")
    aw_num = {"No":0.0,"Sometimes":0.5,"Yes":1.0}[aw]

    if st.button("Predict Risk"):
        Xinp = pd.DataFrame([{"pulling_severity":sev, "awareness_level_encoded":aw_num}])
        pred = label_enc.inverse_transform(model.predict(Xinp))[0]
        st.metric("Estimated Risk Level", pred.title())

    st.markdown("---")
    st.subheader("Popular Coping Strategies")
    if "effective_coping_strategies" in df:
        top = (
            df["effective_coping_strategies"].dropna()
              .str.split(",",expand=True).stack().str.strip()
              .value_counts().head(5)
        )
        st.table(top.rename_axis("Strategy").reset_index(name="Count"))

# ─── EMOTION ASSISTANT TAB ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Emotion Assistant")
    feeling = st.text_input("How are you feeling?", "")
    if st.button("Get Suggestions"):
        txt = feeling.lower()
        if any(k in txt for k in ("stress","anx")):
            sugg = ["Deep breathing","Short walk","Meditation"]
        elif "sad" in txt:
            sugg = ["Uplifting music","Write in journal"]
        else:
            sugg = ["Take a break","Talk to a friend"]
        for s in sugg: st.markdown(f"- {s}")

# ─── TRIGGERS & ENVIRONMENTS TAB ───────────────────────────────────────────────
with tab3:
    st.subheader("Top 10 Emotional Triggers")
    if "pulling_triggers" in df:
        counts = (
            df["pulling_triggers"].dropna()
              .str.split(",",expand=True).stack().str.strip()
              .value_counts().head(10)
        )
        fig = px.bar(counts, title="Top 10 Triggers",
                     labels={"index":"Trigger","value":"Count"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pulling Environment Breakdown")
    if "pulling_environment" in df:
        env = df["pulling_environment"].value_counts()
        fig2 = px.pie(env, names=env.index, values=env.values,
                      title="Where are you pulling?")
        st.plotly_chart(fig2, use_container_width=True)

# ─── BINARY FLAGS TAB ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("Binary-Flag Tables: Yes Counts")
    conn = sqlite3.connect("database/ttm_database.db")
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%\\_1\\_yes\\_0\\_no' ESCAPE '\\'")
    bins = [r[0] for r in cur.fetchall()]
    conn.close()

    for tbl in bins:
        bdf = df[[c for c in df.columns if c.startswith(tbl.replace("_1_yes_0_no",""))]]
        yes = bdf.sum().sort_values(ascending=False)
        fig = px.bar(yes, orientation="h",
                     title=tbl.replace("_1_yes_0_no","").replace("_"," ").title(),
                     labels={"index":"Feature","value":"Yes Count"})
        st.plotly_chart(fig, use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<hr><p style='text-align:center;color:#555;'>© 2025 TrichMind Research • Data remains anonymous</p>",
    unsafe_allow_html=True
)
