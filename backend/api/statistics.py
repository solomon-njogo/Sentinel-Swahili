"""
Statistics API endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from backend.models.schemas import Statistics
from backend.services.data_service import (
    load_all_data, merge_alerts, expand_classification, calculate_statistics
)

router = APIRouter()


@router.get("/statistics", response_model=Statistics)
async def get_statistics(
    data_source: str = Query(default="Both", pattern="^(Agent Reports|Database Alerts|Both)$")
):
    """Get dashboard statistics"""
    try:
        data = load_all_data()
        alerts = merge_alerts(data['agent_alerts_df'], data['db_alerts'], data_source)
        alerts = expand_classification(alerts)
        
        stats = calculate_statistics(
            alerts,
            data['agent_reports'],
            data['db_alerts'],
            data['feedback'],
            data['processed']
        )
        
        return Statistics(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


