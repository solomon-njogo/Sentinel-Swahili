"""
Agent Reports API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List
from backend.models.schemas import AgentReport
from backend.services.data_service import load_all_data

router = APIRouter()


@router.get("/agent-reports", response_model=List[AgentReport])
async def get_agent_reports():
    """Get all agent reports"""
    try:
        data = load_all_data()
        reports = data['agent_reports']
        
        # Convert to response format
        reports_list = []
        for report in reports:
            reports_list.append({
                'report_id': report.get('report_id', ''),
                'raw_message': report.get('raw_message', ''),
                'source': report.get('source'),
                'received_at': report.get('received_at'),
                'processed_at': report.get('processed_at'),
                'status': report.get('status'),
                'validation': report.get('validation', {}),
                'escalation': report.get('escalation', {}),
                'metadata': report.get('metadata', {})
            })
        
        return reports_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-reports/{report_id}", response_model=AgentReport)
async def get_agent_report(report_id: str):
    """Get a single agent report by ID"""
    try:
        data = load_all_data()
        reports = data['agent_reports']
        
        # Find report by ID
        report = next((r for r in reports if r.get('report_id') == report_id), None)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            'report_id': report.get('report_id', ''),
            'raw_message': report.get('raw_message', ''),
            'source': report.get('source'),
            'received_at': report.get('received_at'),
            'processed_at': report.get('processed_at'),
            'status': report.get('status'),
            'validation': report.get('validation', {}),
            'escalation': report.get('escalation', {}),
            'metadata': report.get('metadata', {})
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


