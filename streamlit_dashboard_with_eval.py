#!/usr/bin/env python3
"""
Threat Alert — Streamlit Dashboard + Evaluation Panel
Header:
  Project: Threat Alert — Demo (Streamlit Dashboard)
  Description: Interactive dashboard showing alerts, map, and an evaluation report generator.
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime, timezone
from streamlit_folium import st_folium
import folium
from pathlib import Path

DB = "data/threats.db"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def read_sql_table(db, table):
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def try_parse_json_col(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

def evaluate_all(processed_df, alerts_df, feedback_df):
    # Reuse simplified heuristics from earlier scripts
    report = {}
    report['generated_at'] = datetime.now(timezone.utc).isoformat()
    report['counts'] = {
        'processed_rows': int(processed_df.shape[0]) if processed_df is not None else 0,
        'alerts_rows': int(alerts_df.shape[0]) if alerts_df is not None else 0,
        'feedback_rows': int(feedback_df.shape[0]) if feedback_df is not None else 0
    }
    # Simple numeric scores
    report['scores'] = {}
    report['scores']['concept'] = 10 if report['counts']['processed_rows']>=50 else 6
    report['scores']['methodology'] = 10 if report['counts']['alerts_rows']>0 and report['counts']['feedback_rows']>0 else 7
    report['scores']['technical'] = 12 if 'classification' in alerts_df.columns or 'severity' in alerts_df.columns else 6
    report['scores']['usability'] = 10 if report['counts']['alerts_rows']>0 else 5
    report['scores']['scalability'] = 8 if report['counts']['processed_rows']>=100 else 4
    report['total'] = sum(report['scores'].values())
    return report

st.set_page_config(layout="wide", page_title="Threat Analyst Dashboard (Demo)")

st.title("Threat Analyst Dashboard — Demo")
st.markdown("**Header**: Project: Threat Alert — Demo. Use this dashboard to review alerts, send feedback, and generate evaluation reports.")

# Load data
processed = pd.read_csv("data/processed_reports.csv") if Path("data/processed_reports.csv").exists() else pd.DataFrame()
alerts = read_sql_table(DB, "alerts")
feedback = read_sql_table(DB, "feedback")

# Expand classification if present
if 'classification' in alerts.columns:
    try:
        parsed = alerts['classification'].apply(try_parse_json_col).apply(pd.Series)
        alerts = pd.concat([alerts, parsed], axis=1)
    except Exception:
        pass

left, right = st.columns((1,2))
with left:
    st.header("Filters & Alerts")
    min_sev = st.slider("Minimum severity", 0, 10, 3)
    df_filtered = alerts.copy()
    if 'severity' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['severity'].fillna(0) >= min_sev]
    st.dataframe(df_filtered[['id','text','lat','lon','severity']].fillna(''), height=350)

    st.markdown("### Feedback")
    st.write(feedback)

with right:
    st.header("Map")
    if df_filtered[['lat','lon']].dropna().shape[0] > 0:
        center = (df_filtered['lat'].mean(), df_filtered['lon'].mean())
    else:
        center = (-1.7, 39.86)
    m = folium.Map(location=center, zoom_start=12)
    for _, r in df_filtered.dropna(subset=['lat','lon']).iterrows():
        sev = r.get('severity', None)
        color = 'red' if sev and float(sev) >= 8 else ('orange' if sev and float(sev) >=5 else 'green')
        popup = folium.Popup(f"<b>Text:</b> {r.get('text','')}<br><b>Severity:</b> {sev}", max_width=400)
        folium.CircleMarker(location=(r['lat'], r['lon']), radius=6, color=color, fill=True, popup=popup).add_to(m)
    st_folium(m, width=700, height=450)

st.sidebar.header("Evaluation")
if st.sidebar.button("Run evaluation"):
    report = evaluate_all(processed, alerts, feedback)
    (OUT_DIR / "evaluation_report_streamlit.json").write_text(json.dumps(report, indent=2))
    st.sidebar.success("Evaluation completed and saved to outputs/")
    st.sidebar.json(report)

st.sidebar.markdown("## Summary stats")
st.sidebar.write(f"- Alerts: {len(alerts)}")
st.sidebar.write(f"- Feedback rows: {len(feedback)}")
st.sidebar.write(f"- Processed rows: {len(processed)}")

st.info("Note: Evaluation is heuristic-based and intended to help you iterate quickly. Replace heuristics with ground-truth metrics when available.")
