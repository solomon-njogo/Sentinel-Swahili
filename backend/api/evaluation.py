"""
Evaluation API endpoints
"""

from fastapi import APIRouter, HTTPException
from backend.models.schemas import EvaluationRequest, EvaluationResponse
from backend.services.data_service import load_all_data
from datetime import datetime, timezone
from pathlib import Path
import json

router = APIRouter()

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def evaluate_all(processed_df, alerts_df, feedback_df):
    """Reuse evaluation logic from dashboard"""
    report = {}
    report['generated_at'] = datetime.now(timezone.utc).isoformat()
    report['counts'] = {
        'processed_rows': int(processed_df.shape[0]) if processed_df is not None else 0,
        'alerts_rows': int(alerts_df.shape[0]) if alerts_df is not None else 0,
        'feedback_rows': int(feedback_df.shape[0]) if feedback_df is not None else 0
    }
    # Simple numeric scores
    report['scores'] = {}
    report['scores']['concept'] = 10 if report['counts']['processed_rows'] >= 50 else 6
    report['scores']['methodology'] = 10 if report['counts']['alerts_rows'] > 0 and report['counts']['feedback_rows'] > 0 else 7
    report['scores']['technical'] = 12 if 'classification' in alerts_df.columns or 'severity' in alerts_df.columns else 6
    report['scores']['usability'] = 10 if report['counts']['alerts_rows'] > 0 else 5
    report['scores']['scalability'] = 8 if report['counts']['processed_rows'] >= 100 else 4
    report['total'] = sum(report['scores'].values())
    return report


@router.post("/evaluation", response_model=EvaluationResponse)
async def run_evaluation():
    """Run evaluation and return report"""
    try:
        data = load_all_data()
        report = evaluate_all(
            data['processed'],
            data['agent_alerts_df'],
            data['feedback']
        )
        
        # Save to file
        (OUT_DIR / "evaluation_report_api.json").write_text(json.dumps(report, indent=2))
        
        return EvaluationResponse(**report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


