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
        agent_match = agent_alerts[agent_alerts['id'] == alert_id]
        
        # Search in db alerts
        db_alerts = data['db_alerts']
        db_match = db_alerts[db_alerts['id'] == alert_id]
        
        # Combine and get first match
        if not agent_match.empty:
            row = agent_match.iloc[0]
            source_data = data['agent_reports']
        elif not db_match.empty:
            row = db_match.iloc[0]
            source_data = None
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alerts = expand_classification(pd.DataFrame([row]))
        row = alerts.iloc[0]
        
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
        
        # Add classification data
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
        
        return alert_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


