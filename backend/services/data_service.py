"""
Data service layer - reuses existing dashboard functions
"""

import pandas as pd
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import existing dashboard utilities
try:
    from dashboard.agent_loader import load_agent_reports, get_latest_reports
    from dashboard.location_geocoder import geocode_report_location
except ImportError:
    try:
        from agent_loader import load_agent_reports, get_latest_reports
        from location_geocoder import geocode_report_location
    except ImportError:
        def get_latest_reports(*args, **kwargs):
            return []
        def geocode_report_location(*args, **kwargs):
            return (-6.7924, 39.2083)

DB = "data/threats.db"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def read_sql_table(db: str, table: str) -> pd.DataFrame:
    """Read SQL table into DataFrame"""
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def try_parse_json_col(s):
    """Try to parse JSON column"""
    try:
        return json.loads(s)
    except Exception:
        return {}


def severity_to_numeric(severity_str: str) -> int:
    """Convert severity level string to numeric value"""
    severity_map = {
        "Critical": 10,
        "High": 8,
        "Medium": 5,
        "Low": 2
    }
    return severity_map.get(severity_str, 5)


def numeric_to_severity(severity_num: float) -> str:
    """Convert numeric severity to string"""
    if severity_num >= 9:
        return "Critical"
    elif severity_num >= 7:
        return "High"
    elif severity_num >= 4:
        return "Medium"
    else:
        return "Low"


def convert_agent_reports_to_dataframe(agent_reports: List[Dict]) -> pd.DataFrame:
    """Convert agent reports to DataFrame format"""
    if not agent_reports:
        return pd.DataFrame()
    
    rows = []
    for report in agent_reports:
        try:
            report_id = report.get('report_id', '')
            raw_message = report.get('raw_message', '')
            
            escalation = report.get('escalation', {})
            severity_str = escalation.get('severity', 'Medium')
            severity = severity_to_numeric(severity_str)
            
            lat, lon = geocode_report_location(report)
            
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
        except Exception:
            continue
    
    return pd.DataFrame(rows)


def load_all_data():
    """Load all data sources"""
    # Load agent reports
    agent_reports = get_latest_reports(count=100)
    agent_alerts_df = convert_agent_reports_to_dataframe(agent_reports)
    
    # Load database data
    processed = pd.read_csv("data/processed_reports.csv") if Path("data/processed_reports.csv").exists() else pd.DataFrame()
    db_alerts = read_sql_table(DB, "alerts")
    feedback = read_sql_table(DB, "feedback")
    
    return {
        'agent_reports': agent_reports,
        'agent_alerts_df': agent_alerts_df,
        'db_alerts': db_alerts,
        'processed': processed,
        'feedback': feedback
    }


def merge_alerts(agent_alerts_df: pd.DataFrame, db_alerts: pd.DataFrame, data_source: str) -> pd.DataFrame:
    """Merge alerts based on data source selection"""
    if data_source == "Agent Reports":
        return agent_alerts_df
    elif data_source == "Database Alerts":
        return db_alerts
    else:  # Both
        if not agent_alerts_df.empty and not db_alerts.empty:
            return pd.concat([agent_alerts_df, db_alerts], ignore_index=True)
        elif not agent_alerts_df.empty:
            return agent_alerts_df
        elif not db_alerts.empty:
            return db_alerts
        else:
            return pd.DataFrame()


def expand_classification(alerts: pd.DataFrame) -> pd.DataFrame:
    """Expand classification JSON column if present"""
    if not alerts.empty and 'classification' in alerts.columns:
        try:
            parsed = alerts['classification'].apply(try_parse_json_col).apply(pd.Series)
            alerts = pd.concat([alerts, parsed], axis=1)
        except Exception:
            pass
    return alerts


def filter_alerts(alerts: pd.DataFrame, min_severity: float = 0) -> pd.DataFrame:
    """Filter alerts by minimum severity"""
    df_filtered = alerts.copy()
    if not df_filtered.empty and 'severity' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['severity'].fillna(0) >= min_severity]
    return df_filtered


def calculate_statistics(alerts: pd.DataFrame, agent_reports: List[Dict], 
                        db_alerts: pd.DataFrame, feedback: pd.DataFrame, 
                        processed: pd.DataFrame) -> Dict:
    """Calculate dashboard statistics"""
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
    
    severity_dist = {}
    if not alerts.empty and 'severity' in alerts.columns:
        for _, row in alerts.iterrows():
            sev = numeric_to_severity(row.get('severity', 5))
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
    
    high_severity = severity_dist.get('High', 0) + severity_dist.get('Critical', 0)
    
    return {
        'total_alerts': total_alerts,
        'critical_alerts': critical_alerts,
        'recent_24h': recent_24h,
        'high_priority': high_severity,
        'severity_distribution': severity_dist,
        'agent_reports_count': len(agent_reports),
        'db_alerts_count': len(db_alerts),
        'feedback_count': len(feedback),
        'processed_count': len(processed)
    }


