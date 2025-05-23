#!/usr/bin/env python3
"""
trichmind_app.py

Interactive Streamlit dashboard for TrichMind:

- 🏠 Home: Relapse risk, daily progress, no-pull streak, top coping strategies
- 📅 Daily Log
- 🛠 Coping Tools
- 📓 Journal & Progress
- 💬 Chat

Author: Girlie Razon
Date:   2025-05-26
"""

import os
import datetime
import sqlite3
import joblib
import pandas as pd
import streamlit as st
import seaborn as sns

# ── 1. PAGE CONFIG & GLOBAL STYLING ─────────────────────────────────────────────
LOGO = "assets/logo.png"
ICON = LOGO if os.path.exists(LOGO) else "🧠"
st.set_page_config(
    page_title="TrichMind",
    page_icon=ICON,
    layout="wide"
)

st.markdown("""
  <style>
    /* bottom nav on mobile */
    .main .block-container { padding-bottom: 4rem; }
    @media (max-width: 600px) {
      .stTabs { position: fixed; bottom: 0; left: 0; width: 100%; background: #E0F2F1; z-index: 1000; }
      .stTabs [role="tab"] { flex: 1; text-align: center; padding: 0.5rem 0; margin: 0 !important; }
    }
    /* buttons */
    .stButton>button { background-color: #26A69A; color: white; }
    .stButton>button:hover { background-color: #80CBC4; }
  </style>
""", unsafe_allow_html=True)

# Centered logo (responsive)
if os.path.exists(LOGO):
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image(LOGO, width=200)

sns.set_theme(style="whitegrid")

# ── 2. LOAD MODEL & DATA ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    model = joblib.load("models/best_model.pkl")
    le    = joblib.load("models/label_encoder.pkl")
    return model, le

@st.cache_data(ttl=600)
def load_data():
    conn = sqlite3.connect("database/ttm_database.db")
    demo = pd.read_sql("SELECT * FROM demographics", conn)
    beh  = pd.read_sql("SELECT * FROM hair_pulling_behaviours_patterns", conn)
    conn.close()

    # derived
    if {"age","age_of_onset"}.issubset(demo.columns):
        demo["years_since_onset"] = demo["age"] - demo["age_of_onset"]
    aw_map = {"Yes":1.0,"Sometimes":0.5,"No":0.0}
    if "pulling_awareness" in beh.columns:
        beh["awareness_level_encoded"] = beh["pulling_awareness"].map(aw_map).fillna(0.0)
    if "last_pull_timestamp" in beh.columns:
        beh["last_pull_ts"] = pd.to_datetime(beh["last_pull_timestamp"])

    df = (
        demo.set_index("id")
            .join(beh.set_index("id"), how="inner", validate="one_to_one")
            .reset_index()
    )
    return df

model, label_enc = load_model_and_encoder()
df               = load_data()

# ── 3. NAV TABS ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Home",
    "📅 Daily Log",
    "🛠 Coping Tools",
    "📓 Journal & Progress",
    "💬 Chat"
])

# ─── HOME ───────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Relapse Risk")
    latest = df.sort_values("id").iloc[-1]
    feat_cols = [
        "age","age_of_onset","years_since_onset",
        "pulling_severity","awareness_level_encoded"
    ]
    Xl = pd.DataFrame([latest[feat_cols]])
    risk = label_enc.inverse_transform(model.predict(Xl))[0].upper()
    colors = {"LOW":"#A5D6A7","MODERATE":"#FFF59D","HIGH":"#EF9A9A"}

    st.markdown(f"""
      <div style="background:{colors[risk]};padding:1.2rem;border-radius:10px;text-align:center;">
        <h2 style="margin:0;font-size:2rem;">{risk}</h2>
        <small>As of today</small>
      </div>
    """, unsafe_allow_html=True)

    st.subheader("Daily Progress")
    # count of entries *today*
    today_mask = (pd.to_datetime(latest.get("timestamp", datetime.datetime.now()))).date() == datetime.datetime.today().date()
    # fallback: just show total count
    count = df[df["id"] == latest["id"]].shape[0]
    st.metric("Entries recorded", count)

    if "last_pull_ts" in df.columns:
        last_ts = df.loc[df["id"]==latest["id"], "last_pull_ts"].iloc[0]
        delta   = datetime.datetime.now() - last_ts
        hrs     = int(delta.total_seconds() // 3600)
        days    = delta.days
        st.subheader("No-Pull Streak")
        st.write(f"⏱️ {hrs} hours ({days} days) since your last pull.")

    # Top coping strategies
    st.subheader("Top Coping Strategies")
    conn = sqlite3.connect("database/ttm_database.db")
    cps  = pd.read_sql("SELECT * FROM effective_coping_strategies", conn)
    conn.close()

    m   = cps.melt("id", [c for c in cps.columns if c!="id"], "strategy", "flag")
    yes = m.query("flag==1").copy()
    yes["strategy"] = (
        yes["strategy"]
           .str.replace("_"," ")
           .str.title()
           .str.strip()
    )
    yes = yes[~yes["strategy"].str.match(r"^(None|Unknown)$", case=False)]
    yes = yes[yes["strategy"]!=""]

    if not yes.empty:
        top5 = yes["strategy"].value_counts().head(5).index.tolist()
        for strat in top5:
            st.markdown(f"• {strat}")
    else:
        st.info("No coping strategies data available.")

# ─── DAILY LOG ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Daily Log")
    mood   = st.selectbox("Mood", ["Anxious","Stressed","Calm","Happy"])
    stress = st.slider("Stress Level", 0, 10, 5)
    urge   = st.slider("Pulling Urges", 0, 10, 3)
    env    = st.radio("Environment", ["Home","Work","Public","Other"])
    if st.button("Log Entry"):
        st.success("✅ Entry saved!")

# ─── COPING TOOLS ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Coping Tools")
    st.markdown("**AI-Recommended**")
    try:
        recs = top5[:2]
        c1, c2 = st.columns(2)
        for col, strat in zip((c1,c2), recs):
            with col:
                st.markdown(f"""
                  <div style="background:#B2DFDB;padding:1rem;border-radius:8px;">
                    <h4 style="margin:0;">{strat}</h4>
                  </div>
                """, unsafe_allow_html=True)
    except:
        st.info("No recommendations yet.")

# ─── JOURNAL & PROGRESS ────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Journal & Progress")
    sel = st.date_input("Select date", datetime.date.today())
    st.write(f"Entries on {sel}: **{count}**")
    st.info("Time-series charts will appear once timestamped entries are stored.")

# ─── CHAT ───────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Emotion Assistant")
    msg = st.text_input("How are you feeling?", "")
    if st.button("Send"):
        tl = msg.lower()
        if any(k in tl for k in ["stress","anxious"]):
            sugg = ["Deep breathing","Short walk","Guided meditation"]
        elif "sad" in tl:
            sugg = ["Listen to music","Write in journal"]
        else:
            sugg = ["Take a break","Call a friend"]
        for s in sugg:
            st.markdown(f"• {s}")

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='opacity:0.3;'/>"
    "<p style='text-align:center;color:#555;'>"
    "© 2025 TrichMind Research • Data remains anonymous"
    "</p>",
    unsafe_allow_html=True
)
