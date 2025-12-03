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
from datetime import datetime, timezone, timedelta
from streamlit_folium import st_folium
import folium
from pathlib import Path
import sys
from collections import Counter

# Try to import plotly, fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

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

def numeric_to_severity(severity_num: float) -> str:
    """Convert numeric severity to string."""
    if severity_num >= 9:
        return "Critical"
    elif severity_num >= 7:
        return "High"
    elif severity_num >= 4:
        return "Medium"
    else:
        return "Low"

def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        "Critical": "#D32F2F",
        "High": "#F57C00",
        "Medium": "#FBC02D",
        "Low": "#388E3C"
    }
    return colors.get(severity, "#9CA3AF")

def load_custom_css():
    """Load custom CSS from file."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def create_metric_card(title: str, value: str, change: str = None, change_type: str = "neutral", icon: str = ""):
    """Create a styled metric card."""
    change_html = ""
    if change:
        change_class = "positive" if change_type == "positive" else "negative" if change_type == "negative" else ""
        change_html = f'<div class="metric-card-change {change_class}">{change}</div>'
    
    icon_html = f'<span style="font-size: 1.5rem; margin-bottom: 0.5rem; display: block;">{icon}</span>' if icon else ""
    
    card_html = f"""
    <div class="metric-card fade-in">
        {icon_html}
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value">{value}</div>
        {change_html}
    </div>
    """
    return card_html


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
    """Display agent report details in a user-friendly UI format with modern card design."""
    if not report:
        st.warning("No report data available")
        return
    
    # Report Header Card
    report_id = report.get('report_id', 'Unknown')
    with st.container():
        st.markdown(f"""
        <div class="report-detail-card">
            <h3 style="margin-top: 0; color: #1E88E5;">📋 Report: {report_id}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Metadata section with styled cards
    st.markdown('<div class="report-detail-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Metadata")
    col1, col2, col3 = st.columns(3)
    with col1:
        source = report.get('source', 'N/A')
        st.metric("Source", source)
    with col2:
        status = report.get('status', 'N/A')
        status_badge_class = "status-complete" if status == "complete" else "status-incomplete" if status == "incomplete" else "status-error"
        st.markdown(f'<span class="status-badge {status_badge_class}">{status.title()}</span>', unsafe_allow_html=True)
    with col3:
        language = report.get('metadata', {}).get('language', 'N/A')
        st.metric("Language", language.upper() if language else 'N/A')
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Timestamps
    st.markdown('<div class="report-detail-section">', unsafe_allow_html=True)
    st.markdown("#### ⏰ Timestamps")
    ts_col1, ts_col2 = st.columns(2)
    with ts_col1:
        received = report.get('received_at', 'N/A')
        st.write(f"**Received:** {received}")
    with ts_col2:
        processed = report.get('processed_at', 'N/A')
        st.write(f"**Processed:** {processed}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Raw Message
    st.markdown('<div class="report-detail-section">', unsafe_allow_html=True)
    st.markdown("#### 💬 Raw Message")
    raw_message = report.get('raw_message', 'No message')
    st.info(raw_message)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation Results
    validation = report.get('validation', {})
    if validation:
        st.markdown('<div class="report-detail-section">', unsafe_allow_html=True)
        st.markdown("#### ✅ Validation Results")
        
        # Overall completeness with progress bar
        completeness = validation.get('overall_completeness', 0.0)
        val_status = validation.get('status', 'unknown')
        
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.metric("Overall Completeness", f"{completeness:.1%}")
            # Progress bar
            progress_color = "progress-success" if completeness >= 0.8 else "progress-medium" if completeness >= 0.5 else "progress-critical"
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar {progress_color}" style="width: {completeness * 100}%"></div>
            </div>
            """, unsafe_allow_html=True)
        with val_col2:
            status_badge_class = "status-complete" if val_status == "complete" else "status-incomplete" if val_status == "incomplete" else "status-error"
            st.markdown(f'<span class="status-badge {status_badge_class}">{val_status.title()}</span>', unsafe_allow_html=True)
        
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
        
        # Field Scores with progress bars
        field_scores = validation.get('field_scores', [])
        if field_scores:
            st.markdown("**Field Completeness Scores:**")
            for field_score in field_scores:
                field_name = field_score.get('field_name', 'unknown')
                score = field_score.get('score', 0.0)
                is_present = field_score.get('is_present', False)
                confidence = field_score.get('confidence', 0.0)
                
                progress_color = "progress-success" if score >= 0.8 else "progress-medium" if score >= 0.5 else "progress-critical"
                status_icon = "✅" if is_present else "❌"
                
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <strong>{field_name.title()}</strong>
                        <span>{score:.1%} {status_icon}</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar {progress_color}" style="width: {score * 100}%"></div>
                    </div>
                    <small style="color: #6B7280;">Confidence: {confidence:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Escalation Results
    escalation = report.get('escalation', {})
    if escalation:
        st.markdown('<div class="report-detail-section">', unsafe_allow_html=True)
        st.markdown("#### 🚨 Escalation Results")
        
        severity = escalation.get('severity', 'Unknown')
        severity_class = f"severity-{severity.lower()}" if severity in ["Critical", "High", "Medium", "Low"] else ""
        
        esc_col1, esc_col2, esc_col3 = st.columns(3)
        
        with esc_col1:
            st.markdown(f'<span class="severity-badge {severity_class}">{severity}</span>', unsafe_allow_html=True)
            st.metric("Severity Level", severity)
        
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
            keyword_html = " ".join([f'<span class="keyword-tag">{kw}</span>' for kw in urgency_keywords])
            st.markdown(keyword_html, unsafe_allow_html=True)
        
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
        
        st.markdown('</div>', unsafe_allow_html=True)
    
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

st.set_page_config(
    layout="wide", 
    page_title="Threat Alert Dashboard",
    page_icon="🚨",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Header Section
st.markdown("""
<div style="padding: 1rem 0 2rem 0; border-bottom: 2px solid #E5E7EB; margin-bottom: 2rem;">
    <h1 style="margin: 0; color: #1E88E5; font-size: 2.5rem; font-weight: 700;">🚨 Threat Alert Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; color: #6B7280; font-size: 1rem;">Real-time threat intelligence and monitoring system</p>
</div>
""", unsafe_allow_html=True)

# Load agent reports
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_agent_reports_cached():
    """Load agent reports with caching."""
    try:
        return get_latest_reports(count=100)
    except Exception as e:
        # Return empty list on error (can't use st.warning in cached function)
        return []

with st.spinner("Loading threat reports..."):
    agent_reports = load_agent_reports_cached()
    
if not agent_reports:
    st.sidebar.warning("No agent reports found. Check agents/storage/ directory.")
agent_alerts_df = convert_agent_reports_to_dataframe(agent_reports)

# Load data from database
processed = pd.read_csv("data/processed_reports.csv") if Path("data/processed_reports.csv").exists() else pd.DataFrame()
db_alerts = read_sql_table(DB, "alerts")
feedback = read_sql_table(DB, "feedback")

# Sidebar - Data Source Selection
with st.sidebar:
    st.markdown("### 📊 Data Source")
    data_source = st.radio(
        "Select data source:",
        ["Agent Reports", "Database Alerts", "Both"],
        index=0,
        key="data_source_selector",
        label_visibility="collapsed"
    )
    st.divider()

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

# Calculate metrics for header cards
total_alerts = len(alerts)
critical_alerts = len(alerts[alerts.get('severity', 0) >= 9]) if not alerts.empty and 'severity' in alerts.columns else 0
recent_24h = 0
if not alerts.empty and 'received_at' in alerts.columns:
    try:
        now = datetime.now(timezone.utc)
        alerts['received_at_parsed'] = pd.to_datetime(alerts['received_at'], errors='coerce')
        recent_24h = len(alerts[alerts['received_at_parsed'] >= now - timedelta(hours=24)])
    except:
        pass

# Key Metrics Cards
st.markdown("### 📈 Overview")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown(create_metric_card("Total Alerts", str(total_alerts), icon="📊"), unsafe_allow_html=True)

with metric_col2:
    st.markdown(create_metric_card("Critical Alerts", str(critical_alerts), icon="🔴"), unsafe_allow_html=True)

with metric_col3:
    st.markdown(create_metric_card("Last 24 Hours", str(recent_24h), icon="⏰"), unsafe_allow_html=True)

with metric_col4:
    severity_dist = {}
    if not alerts.empty and 'severity' in alerts.columns:
        for _, row in alerts.iterrows():
            sev = numeric_to_severity(row.get('severity', 5))
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
    high_severity = severity_dist.get('High', 0) + severity_dist.get('Critical', 0)
    st.markdown(create_metric_card("High Priority", str(high_severity), icon="⚠️"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Content Area
left, right = st.columns((1, 2))

with left:
    st.markdown('<div class="section-header">🔍 Filters & Alerts</div>', unsafe_allow_html=True)
    
    # Enhanced filter section
    with st.container():
        st.markdown("**Severity Filter**")
        min_sev = st.slider(
            "Minimum severity level",
            0, 10, 3,
            key="severity_slider",
            help="Filter alerts by minimum severity level (0-10)"
        )
        
        # Show severity breakdown
        if not alerts.empty and 'severity' in alerts.columns:
            st.markdown("**Severity Distribution**")
            severity_counts = {
                'Critical': len(alerts[alerts['severity'] >= 9]),
                'High': len(alerts[(alerts['severity'] >= 7) & (alerts['severity'] < 9)]),
                'Medium': len(alerts[(alerts['severity'] >= 4) & (alerts['severity'] < 7)]),
                'Low': len(alerts[alerts['severity'] < 4])
            }
            for sev, count in severity_counts.items():
                if count > 0:
                    color = get_severity_color(sev)
                    st.markdown(f'<span style="color: {color}; font-weight: 600;">{sev}:</span> {count}', unsafe_allow_html=True)
    df_filtered = alerts.copy()
    if not df_filtered.empty and 'severity' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['severity'].fillna(0) >= min_sev]
    
    # Display filtered alerts with styling
    if not df_filtered.empty:
        st.markdown(f"**Showing {len(df_filtered)} alert(s)**")
        display_cols = ['id', 'text', 'severity']
        # Add source column if available
        if 'source' in df_filtered.columns:
            display_cols.append('source')
        
        # Create styled dataframe
        display_df = df_filtered[display_cols].copy() if all(col in df_filtered.columns for col in display_cols) else df_filtered.copy()
        
        # Add severity labels
        if 'severity' in display_df.columns:
            display_df['Severity'] = display_df['severity'].apply(numeric_to_severity)
            display_df = display_df.drop('severity', axis=1)
        
        # Truncate text for display
        if 'text' in display_df.columns:
            display_df['text'] = display_df['text'].apply(lambda x: x[:100] + "..." if isinstance(x, str) and len(x) > 100 else x)
        
        st.dataframe(
            display_df.fillna(''),
            height=350,
            use_container_width=True,
            hide_index=True
        )
        
        # Show agent report details if selected
        if data_source in ["Agent Reports", "Both"] and not agent_alerts_df.empty and 'source' in df_filtered.columns:
            agent_ids = df_filtered[df_filtered['source'] == 'agent']['id'].tolist()
            if agent_ids:
                st.divider()
                st.markdown("### 📋 Report Details")
                selected_ids = st.multiselect(
                    "Select reports to view:",
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
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📭</div>
            <p>No alerts to display</p>
            <p style="font-size: 0.875rem;">Try selecting a different data source or adjust filters.</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-header">🗺️ Threat Map</div>', unsafe_allow_html=True)
    
    # Map container
    map_col1, map_col2 = st.columns([3, 1])
    
    with map_col2:
        st.markdown("**Map Legend**")
        st.markdown('<span style="color: #D32F2F;">🔴 Critical</span>', unsafe_allow_html=True)
        st.markdown('<span style="color: #F57C00;">🟠 High</span>', unsafe_allow_html=True)
        st.markdown('<span style="color: #FBC02D;">🟡 Medium</span>', unsafe_allow_html=True)
        st.markdown('<span style="color: #388E3C;">🟢 Low</span>', unsafe_allow_html=True)
    
    with map_col1:
        if not df_filtered.empty and df_filtered[['lat','lon']].dropna().shape[0] > 0:
            center = (df_filtered['lat'].mean(), df_filtered['lon'].mean())
            zoom = 12
        else:
            # Default to Dar es Salaam
            center = (-6.7924, 39.2083)
            zoom = 10
        
        # Create map with better styling
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='OpenStreetMap',
            attr='Threat Alert Dashboard'
        )
        
        if not df_filtered.empty:
            for _, r in df_filtered.dropna(subset=['lat','lon']).iterrows():
                sev = r.get('severity', None)
                sev_float = float(sev) if sev is not None else 0
                severity_label = numeric_to_severity(sev_float)
                color = get_severity_color(severity_label)
                
                # Enhanced popup with better styling
                popup_html = f"""
                <div style="font-family: 'Inter', sans-serif; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0; color: {color}; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                        {severity_label} Alert
                    </h4>
                    <p style="margin: 5px 0;"><strong>ID:</strong> {r.get('id', 'N/A')}</p>
                    <p style="margin: 5px 0;"><strong>Severity:</strong> {sev_float}/10</p>
                    <p style="margin: 5px 0; font-size: 0.9em; color: #666;">
                        {r.get('text', '')[:150]}{'...' if len(str(r.get('text', ''))) > 150 else ''}
                    </p>
                """
                if 'source' in r and r['source'] == 'agent':
                    popup_html += f'<p style="margin: 5px 0;"><strong>Source:</strong> Agent Report</p>'
                    if 'priority_score' in r:
                        popup_html += f'<p style="margin: 5px 0;"><strong>Priority:</strong> {r["priority_score"]:.2f}</p>'
                popup_html += "</div>"
                
                popup = folium.Popup(popup_html, max_width=300)
                
                # Enhanced marker styling
                folium.CircleMarker(
                    location=(r['lat'], r['lon']),
                    radius=10 if sev_float >= 8 else 8 if sev_float >= 5 else 6,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2,
                    popup=popup,
                    tooltip=f"{severity_label}: {r.get('id', 'N/A')}"
                ).add_to(m)
        
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st_folium(m, width=None, height=500, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Severity Distribution Chart
    if not alerts.empty and 'severity' in alerts.columns and PLOTLY_AVAILABLE:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Severity Distribution")
        
        severity_data = []
        for _, row in alerts.iterrows():
            sev = numeric_to_severity(row.get('severity', 5))
            severity_data.append(sev)
        
        if severity_data:
            severity_counts = Counter(severity_data)
            fig = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                color=list(severity_counts.keys()),
                color_discrete_map={
                    'Critical': '#D32F2F',
                    'High': '#F57C00',
                    'Medium': '#FBC02D',
                    'Low': '#388E3C'
                },
                hole=0.4
            )
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(family="Inter, sans-serif")
            )
            st.plotly_chart(fig, use_container_width=True)
    elif not alerts.empty and 'severity' in alerts.columns and not PLOTLY_AVAILABLE:
        # Fallback: show text-based distribution
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Severity Distribution")
        severity_data = []
        for _, row in alerts.iterrows():
            sev = numeric_to_severity(row.get('severity', 5))
            severity_data.append(sev)
        if severity_data:
            severity_counts = Counter(severity_data)
            for sev, count in sorted(severity_counts.items(), key=lambda x: x[1], reverse=True):
                color = get_severity_color(sev)
                percentage = (count / len(severity_data)) * 100
                st.markdown(f'<span style="color: {color}; font-weight: 600;">{sev}:</span> {count} ({percentage:.1f}%)', unsafe_allow_html=True)

# Sidebar - Statistics and Evaluation
with st.sidebar:
    st.markdown("### 📈 Statistics")
    
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric("Total Alerts", len(alerts))
    with stats_col2:
        st.metric("Filtered", len(df_filtered) if not df_filtered.empty else 0)
    
    if data_source in ["Agent Reports", "Both"]:
        st.metric("Agent Reports", len(agent_alerts_df))
    if data_source in ["Database Alerts", "Both"]:
        st.metric("DB Alerts", len(db_alerts))
    
    st.metric("Feedback", len(feedback))
    st.metric("Processed", len(processed))
    
    st.divider()
    
    # Agent report statistics
    if data_source in ["Agent Reports", "Both"] and agent_reports:
        st.markdown("### 📊 Severity Breakdown")
        severity_counts = {}
        for report in agent_reports:
            sev = report.get('escalation', {}).get('severity', 'Unknown')
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        for sev, count in sorted(severity_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                color = get_severity_color(sev)
                st.markdown(f'<span style="color: {color}; font-weight: 600;">{sev}:</span> {count}', unsafe_allow_html=True)
    
    st.divider()
    
    # Evaluation Section
    st.markdown("### 🔬 Evaluation")
    if st.button("Run Evaluation", key="run_evaluation_btn", use_container_width=True):
        with st.spinner("Running evaluation..."):
            report = evaluate_all(processed, alerts, feedback)
            (OUT_DIR / "evaluation_report_streamlit.json").write_text(json.dumps(report, indent=2))
            st.success("✅ Evaluation completed!")
            with st.expander("View Report"):
                st.json(report)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6B7280; font-size: 0.875rem; border-top: 1px solid #E5E7EB; margin-top: 2rem;">
    <p style="margin: 0;">Threat Alert Dashboard | Real-time threat intelligence monitoring</p>
    <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem;">Note: Evaluation is heuristic-based. Replace with ground-truth metrics when available.</p>
</div>
""", unsafe_allow_html=True)