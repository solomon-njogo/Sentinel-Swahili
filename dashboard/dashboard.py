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
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import agent loader and geocoder
try:
    from dashboard.agent_loader import load_agent_reports, get_latest_reports
    from dashboard.location_geocoder import geocode_report_location
except ImportError:
    # Fallback: try relative imports if running from dashboard directory
    try:
        from agent_loader import load_agent_reports, get_latest_reports
        from location_geocoder import geocode_report_location
    except ImportError:
        # If both fail, create stub functions
        def get_latest_reports(*args, **kwargs):
            return []
        def geocode_report_location(*args, **kwargs):
            return (-6.7924, 39.2083)  # Default to Dar es Salaam

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

def severity_to_numeric(severity_str: str) -> int:
    """Convert severity level string to numeric value."""
    severity_map = {
        "Critical": 10,
        "High": 8,
        "Medium": 5,
        "Low": 2
    }
    return severity_map.get(severity_str, 5)


def convert_agent_reports_to_dataframe(agent_reports: list) -> pd.DataFrame:
    """Convert agent reports to DataFrame format compatible with dashboard."""
    if not agent_reports:
        return pd.DataFrame()
    
    rows = []
    for report in agent_reports:
        try:
            # Extract data from agent report
            report_id = report.get('report_id', '')
            raw_message = report.get('raw_message', '')
            
            # Get severity and convert to numeric
            escalation = report.get('escalation', {})
            severity_str = escalation.get('severity', 'Medium')
            severity = severity_to_numeric(severity_str)
            
            # Geocode location
            lat, lon = geocode_report_location(report)
            
            # Create classification JSON with validation and escalation data
            classification = {
                'validation': report.get('validation', {}),
                'escalation': escalation,
                'metadata': report.get('metadata', {})
            }
            
            rows.append({
                'id': report_id,
                'text': raw_message,
                'lat': lat,
                'lon': lon,
                'severity': severity,
                'classification': json.dumps(classification),
                'source': 'agent',
                'report_id': report_id,
                'received_at': report.get('received_at', ''),
                'processed_at': report.get('processed_at', ''),
                'priority_score': escalation.get('priority_score', 0.0),
                'requires_immediate_alert': escalation.get('requires_immediate_alert', False)
            })
        except Exception as e:
            # Skip reports that fail to convert
            continue
    
    return pd.DataFrame(rows)


def display_agent_report_details(report: dict):
    """Display agent report details in a user-friendly UI format."""
    if not report:
        st.warning("No report data available")
        return
    
    # Report Header
    report_id = report.get('report_id', 'Unknown')
    st.markdown(f"### 📋 Report: {report_id}")
    
    # Metadata section
    col1, col2, col3 = st.columns(3)
    with col1:
        source = report.get('source', 'N/A')
        st.metric("Source", source)
    with col2:
        status = report.get('status', 'N/A')
        st.metric("Status", status)
    with col3:
        language = report.get('metadata', {}).get('language', 'N/A')
        st.metric("Language", language.upper() if language else 'N/A')
    
    # Timestamps
    st.markdown("#### ⏰ Timestamps")
    ts_col1, ts_col2 = st.columns(2)
    with ts_col1:
        received = report.get('received_at', 'N/A')
        st.write(f"**Received:** {received}")
    with ts_col2:
        processed = report.get('processed_at', 'N/A')
        st.write(f"**Processed:** {processed}")
    
    # Raw Message
    st.markdown("#### 💬 Raw Message")
    raw_message = report.get('raw_message', 'No message')
    st.info(raw_message)
    
    # Validation Results
    validation = report.get('validation', {})
    if validation:
        st.markdown("#### ✅ Validation Results")
        
        # Overall completeness
        completeness = validation.get('overall_completeness', 0.0)
        val_status = validation.get('status', 'unknown')
        
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.metric("Overall Completeness", f"{completeness:.1%}")
        with val_col2:
            status_color = "🟢" if val_status == "complete" else "🟡" if val_status == "incomplete" else "🔴"
            st.metric("Status", f"{status_color} {val_status.title()}")
        
        # Extracted Entities
        entities = validation.get('entities', {})
        if entities:
            st.markdown("**Extracted Entities:**")
            
            entity_col1, entity_col2 = st.columns(2)
            
            with entity_col1:
                st.markdown("**👤 Who:**")
                who_list = entities.get('who', [])
                if who_list:
                    for who in who_list[:5]:  # Limit to 5 items
                        st.write(f"  • {who}")
                    if len(who_list) > 5:
                        st.caption(f"... and {len(who_list) - 5} more")
                else:
                    st.write("  *No data*")
                
                st.markdown("**📍 Where:**")
                where_list = entities.get('where', [])
                if where_list:
                    for where in where_list[:5]:
                        st.write(f"  • {where}")
                    if len(where_list) > 5:
                        st.caption(f"... and {len(where_list) - 5} more")
                else:
                    st.write("  *No data*")
            
            with entity_col2:
                st.markdown("**📝 What:**")
                what_list = entities.get('what', [])
                if what_list:
                    # Show first item (usually most complete)
                    st.write(f"  {what_list[0][:200]}{'...' if len(what_list[0]) > 200 else ''}")
                    if len(what_list) > 1:
                        st.caption(f"... and {len(what_list) - 1} more variations")
                else:
                    st.write("  *No data*")
                
                st.markdown("**🕐 When:**")
                when_list = entities.get('when', [])
                if when_list:
                    for when in when_list[:5]:
                        st.write(f"  • {when}")
                    if len(when_list) > 5:
                        st.caption(f"... and {len(when_list) - 5} more")
                else:
                    st.write("  *No data*")
        
        # Field Scores
        field_scores = validation.get('field_scores', [])
        if field_scores:
            st.markdown("**Field Completeness Scores:**")
            for field_score in field_scores:
                field_name = field_score.get('field_name', 'unknown')
                score = field_score.get('score', 0.0)
                is_present = field_score.get('is_present', False)
                confidence = field_score.get('confidence', 0.0)
                
                score_color = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                st.write(f"{score_color} **{field_name.title()}**: {score:.1%} (Confidence: {confidence:.1%}, Present: {'Yes' if is_present else 'No'})")
    
    # Escalation Results
    escalation = report.get('escalation', {})
    if escalation:
        st.markdown("#### 🚨 Escalation Results")
        
        esc_col1, esc_col2, esc_col3 = st.columns(3)
        
        severity = escalation.get('severity', 'Unknown')
        severity_emoji = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(severity, '⚪')
        
        with esc_col1:
            st.metric("Severity", f"{severity_emoji} {severity}")
        
        priority_score = escalation.get('priority_score', 0.0)
        with esc_col2:
            st.metric("Priority Score", f"{priority_score:.2f}")
        
        confidence = escalation.get('classification_confidence', 0.0)
        with esc_col3:
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Urgency keywords
        urgency_keywords = escalation.get('urgency_keywords_found', [])
        if urgency_keywords:
            st.markdown("**Urgency Keywords Found:**")
            keyword_tags = " ".join([f"`{kw}`" for kw in urgency_keywords])
            st.markdown(keyword_tags)
        
        # Escalation window
        window_minutes = escalation.get('escalation_window_minutes', 0)
        if window_minutes:
            if window_minutes < 60:
                window_display = f"{window_minutes} minutes"
            elif window_minutes < 1440:
                window_display = f"{window_minutes // 60} hours"
            else:
                window_display = f"{window_minutes // 1440} days"
            st.write(f"**Escalation Window:** {window_display}")
        
        # Immediate alert requirement
        requires_alert = escalation.get('requires_immediate_alert', False)
        if requires_alert:
            st.warning("⚠️ **Requires Immediate Alert**")
    
    # Run metadata
    run_number = report.get('run_number')
    report_number = report.get('report_number_in_run')
    if run_number is not None:
        st.markdown("#### 📊 Run Information")
        st.write(f"**Run Number:** {run_number}")
        if report_number is not None:
            st.write(f"**Report Number in Run:** {report_number}")
        log_file = report.get('log_file')
        if log_file:
            st.write(f"**Log File:** {log_file}")
    
    # Expandable raw JSON view
    with st.expander("🔧 View Raw JSON"):
        st.json(report)


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

# Load agent reports
@st.cache_data
def load_agent_reports_cached():
    """Load agent reports with caching."""
    try:
        return get_latest_reports(count=100)
    except Exception as e:
        # Return empty list on error (can't use st.warning in cached function)
        return []

agent_reports = load_agent_reports_cached()
if not agent_reports:
    st.sidebar.warning("No agent reports found. Check agents/storage/ directory.")
agent_alerts_df = convert_agent_reports_to_dataframe(agent_reports)

# Load data from database
processed = pd.read_csv("data/processed_reports.csv") if Path("data/processed_reports.csv").exists() else pd.DataFrame()
db_alerts = read_sql_table(DB, "alerts")
feedback = read_sql_table(DB, "feedback")

# Data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Select data source:",
    ["Agent Reports", "Database Alerts", "Both"],
    index=0,
    key="data_source_selector"
)

# Merge data sources based on selection
if data_source == "Agent Reports":
    alerts = agent_alerts_df
elif data_source == "Database Alerts":
    alerts = db_alerts
else:  # Both
    # Combine both dataframes
    if not agent_alerts_df.empty and not db_alerts.empty:
        alerts = pd.concat([agent_alerts_df, db_alerts], ignore_index=True)
    elif not agent_alerts_df.empty:
        alerts = agent_alerts_df
    elif not db_alerts.empty:
        alerts = db_alerts
    else:
        alerts = pd.DataFrame()

# Expand classification if present
if not alerts.empty and 'classification' in alerts.columns:
    try:
        parsed = alerts['classification'].apply(try_parse_json_col).apply(pd.Series)
        alerts = pd.concat([alerts, parsed], axis=1)
    except Exception:
        pass

left, right = st.columns((1,2))
with left:
    st.header("Filters & Alerts")
    min_sev = st.slider("Minimum severity", 0, 10, 3, key="severity_slider")
    df_filtered = alerts.copy()
    if not df_filtered.empty and 'severity' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['severity'].fillna(0) >= min_sev]
    
    # Display filtered alerts
    if not df_filtered.empty:
        display_cols = ['id', 'text', 'lat', 'lon', 'severity']
        # Add source column if available
        if 'source' in df_filtered.columns:
            display_cols.append('source')
        available_cols = [col for col in display_cols if col in df_filtered.columns]
        st.dataframe(df_filtered[available_cols].fillna(''), height=350)
        
        # Show agent report details if selected
        if data_source in ["Agent Reports", "Both"] and not agent_alerts_df.empty and 'source' in df_filtered.columns:
            agent_ids = df_filtered[df_filtered['source'] == 'agent']['id'].tolist()
            if agent_ids:
                st.markdown("### Agent Report Details")
                selected_ids = st.multiselect(
                    "Select reports to view details:",
                    options=agent_ids,
                    default=[],
                    key="report_details_selector"
                )
                for report_id in selected_ids:
                    report = next((r for r in agent_reports if r.get('report_id') == report_id), None)
                    if report:
                        display_agent_report_details(report)
                        st.divider()
    else:
        st.info("No alerts to display. Try selecting a different data source or adjust filters.")

    st.markdown("### Feedback")
    st.write(feedback)

with right:
    st.header("Map")
    if not df_filtered.empty and df_filtered[['lat','lon']].dropna().shape[0] > 0:
        center = (df_filtered['lat'].mean(), df_filtered['lon'].mean())
        zoom = 12
    else:
        # Default to Dar es Salaam
        center = (-6.7924, 39.2083)
        zoom = 10
    
    m = folium.Map(location=center, zoom_start=zoom)
    
    if not df_filtered.empty:
        for _, r in df_filtered.dropna(subset=['lat','lon']).iterrows():
            sev = r.get('severity', None)
            sev_float = float(sev) if sev is not None else 0
            
            # Color coding: red for critical/high, orange for medium, green for low
            if sev_float >= 8:
                color = 'red'
            elif sev_float >= 5:
                color = 'orange'
            else:
                color = 'green'
            
            # Build popup content
            popup_text = f"<b>ID:</b> {r.get('id', 'N/A')}<br>"
            popup_text += f"<b>Text:</b> {r.get('text', '')[:100]}...<br>"
            popup_text += f"<b>Severity:</b> {sev}<br>"
            if 'source' in r and r['source'] == 'agent':
                popup_text += f"<b>Source:</b> Agent Report<br>"
                if 'priority_score' in r:
                    popup_text += f"<b>Priority Score:</b> {r['priority_score']:.2f}<br>"
            
            popup = folium.Popup(popup_text, max_width=400)
            folium.CircleMarker(
                location=(r['lat'], r['lon']),
                radius=8 if sev_float >= 8 else 6,
                color=color,
                fill=True,
                fillOpacity=0.7,
                popup=popup
            ).add_to(m)
    
    st_folium(m, width=700, height=450)

st.sidebar.header("Evaluation")
if st.sidebar.button("Run evaluation", key="run_evaluation_btn"):
    report = evaluate_all(processed, alerts, feedback)
    (OUT_DIR / "evaluation_report_streamlit.json").write_text(json.dumps(report, indent=2))
    st.sidebar.success("Evaluation completed and saved to outputs/")
    st.sidebar.json(report)

st.sidebar.markdown("## Summary stats")
st.sidebar.write(f"- Total Alerts: {len(alerts)}")
if data_source in ["Agent Reports", "Both"]:
    st.sidebar.write(f"- Agent Reports: {len(agent_alerts_df)}")
if data_source in ["Database Alerts", "Both"]:
    st.sidebar.write(f"- Database Alerts: {len(db_alerts)}")
st.sidebar.write(f"- Feedback rows: {len(feedback)}")
st.sidebar.write(f"- Processed rows: {len(processed)}")

# Agent report statistics
if data_source in ["Agent Reports", "Both"] and agent_reports:
    st.sidebar.markdown("## Agent Report Stats")
    severity_counts = {}
    for report in agent_reports:
        sev = report.get('escalation', {}).get('severity', 'Unknown')
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    for sev, count in sorted(severity_counts.items(), key=lambda x: x[1], reverse=True):
        st.sidebar.write(f"- {sev}: {count}")

st.info("Note: Evaluation is heuristic-based and intended to help you iterate quickly. Replace heuristics with ground-truth metrics when available.")