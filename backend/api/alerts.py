"""
Alert API endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from backend.models.schemas import Alert, AlertFilter
from backend.services.data_service import (
    load_all_data, merge_alerts, expand_classification, 
    filter_alerts, numeric_to_severity
)
import pandas as pd

router = APIRouter()


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    min_severity: float = Query(default=0, ge=0, le=10),
    data_source: str = Query(default="Both", pattern="^(Agent Reports|Database Alerts|Both)$")
):
    """Get all alerts with optional filtering"""
    try:
        data = load_all_data()
        alerts = merge_alerts(data['agent_alerts_df'], data['db_alerts'], data_source)
        alerts = expand_classification(alerts)
        alerts = filter_alerts(alerts, min_severity)
        
        # Convert to list of dicts
        alerts_list = []
        for _, row in alerts.iterrows():
            alert_dict = {
                'id': row.get('id', ''),
                'text': row.get('text', ''),
                'severity': float(row.get('severity', 0)) if pd.notna(row.get('severity')) else None,
                'lat': float(row.get('lat')) if pd.notna(row.get('lat')) else None,
                'lon': float(row.get('lon')) if pd.notna(row.get('lon')) else None,
                'source': row.get('source'),
                'received_at': row.get('received_at'),
                'processed_at': row.get('processed_at'),
            }
            
            # Add classification data if available
            if 'validation' in row:
                alert_dict['validation'] = row['validation'] if isinstance(row['validation'], dict) else {}
            if 'escalation' in row:
                alert_dict['escalation'] = row['escalation'] if isinstance(row['escalation'], dict) else {}
            if 'classification' in row:
                alert_dict['classification'] = row['classification'] if isinstance(row['classification'], dict) else {}
            if 'priority_score' in row:
                alert_dict['priority_score'] = float(row['priority_score']) if pd.notna(row.get('priority_score')) else None
            if 'requires_immediate_alert' in row:
                alert_dict['requires_immediate_alert'] = bool(row['requires_immediate_alert']) if pd.notna(row.get('requires_immediate_alert')) else None
            
            alerts_list.append(alert_dict)
        
        return alerts_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{alert_id}", response_model=Alert)
async def get_alert(alert_id: str):
    """Get a single alert by ID"""
    try:
        data = load_all_data()
        
        # Search in agent alerts
        agent_alerts = data['agent_alerts_df']
        agent_match = None
        if not agent_alerts.empty and 'id' in agent_alerts.columns:
            agent_match = agent_alerts[agent_alerts['id'] == alert_id]
        
        # Search in db alerts
        db_alerts = data['db_alerts']
        db_match = None
        if not db_alerts.empty and 'id' in db_alerts.columns:
            db_match = db_alerts[db_alerts['id'] == alert_id]
        
        # Combine and get first match
        row = None
        if agent_match is not None and not agent_match.empty:
            row = agent_match.iloc[0]
        elif db_match is not None and not db_match.empty:
            row = db_match.iloc[0]
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        # Convert row to dict first, then create DataFrame
        row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        alerts_df = pd.DataFrame([row_dict])
        alerts = expand_classification(alerts_df)
        processed_row = alerts.iloc[0]
        
        # Helper function to safely get value from pandas Series
        def safe_get(series, key, default=None):
            if key not in series.index:
                return default
            val = series[key]
            return default if pd.isna(val) else val
        
        # Convert to dict safely
        alert_dict = {
            'id': str(safe_get(processed_row, 'id', '')),
            'text': str(safe_get(processed_row, 'text', '')),
            'severity': float(processed_row['severity']) if 'severity' in processed_row.index and pd.notna(processed_row['severity']) else None,
            'lat': float(processed_row['lat']) if 'lat' in processed_row.index and pd.notna(processed_row['lat']) else None,
            'lon': float(processed_row['lon']) if 'lon' in processed_row.index and pd.notna(processed_row['lon']) else None,
            'source': str(processed_row['source']) if 'source' in processed_row.index and pd.notna(processed_row['source']) else None,
            'received_at': str(processed_row['received_at']) if 'received_at' in processed_row.index and pd.notna(processed_row['received_at']) else None,
            'processed_at': str(processed_row['processed_at']) if 'processed_at' in processed_row.index and pd.notna(processed_row['processed_at']) else None,
        }
        
        # Add classification data
        if 'validation' in processed_row.index and pd.notna(processed_row['validation']):
            val = processed_row['validation']
            alert_dict['validation'] = val if isinstance(val, dict) else {}
        if 'escalation' in processed_row.index and pd.notna(processed_row['escalation']):
            esc = processed_row['escalation']
            alert_dict['escalation'] = esc if isinstance(esc, dict) else {}
        if 'classification' in processed_row.index and pd.notna(processed_row['classification']):
            cls = processed_row['classification']
            alert_dict['classification'] = cls if isinstance(cls, dict) else {}
        if 'priority_score' in processed_row.index and pd.notna(processed_row['priority_score']):
            alert_dict['priority_score'] = float(processed_row['priority_score'])
        if 'requires_immediate_alert' in processed_row.index and pd.notna(processed_row['requires_immediate_alert']):
            alert_dict['requires_immediate_alert'] = bool(processed_row['requires_immediate_alert'])
        
        return alert_dict
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


