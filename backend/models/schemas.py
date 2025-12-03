"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AlertBase(BaseModel):
    """Base alert model"""
    id: str
    text: str
    severity: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    source: Optional[str] = None
    received_at: Optional[str] = None
    processed_at: Optional[str] = None


class Alert(AlertBase):
    """Alert response model"""
    classification: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    escalation: Optional[Dict[str, Any]] = None
    priority_score: Optional[float] = None
    requires_immediate_alert: Optional[bool] = None

    class Config:
        from_attributes = True
        extra = "allow"  # Allow extra fields in reports


class AgentReport(BaseModel):
    """Agent report model"""
    report_id: str
    raw_message: str
    source: Optional[str] = None
    received_at: Optional[str] = None
    processed_at: Optional[str] = None
    status: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    escalation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Additional fields from full report
    run_number: Optional[int] = None
    report_number_in_run: Optional[int] = None
    log_file: Optional[str] = None
    collected_where: Optional[str] = None
    collected_what: Optional[str] = None
    collected_who: Optional[str] = None
    collected_when: Optional[str] = None
    conversation_flow: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
        extra = "allow"  # Allow extra fields in reports


class Statistics(BaseModel):
    """Dashboard statistics model"""
    total_alerts: int
    critical_alerts: int
    recent_24h: int
    high_priority: int
    severity_distribution: Dict[str, int]
    agent_reports_count: int
    db_alerts_count: int
    feedback_count: int
    processed_count: int


class EvaluationRequest(BaseModel):
    """Evaluation request model"""
    processed_count: int = 0
    alerts_count: int = 0
    feedback_count: int = 0


class EvaluationResponse(BaseModel):
    """Evaluation response model"""
    generated_at: str
    counts: Dict[str, int]
    scores: Dict[str, int]
    total: int


class AlertFilter(BaseModel):
    """Alert filter model"""
    min_severity: Optional[float] = Field(default=0, ge=0, le=10)
    source: Optional[str] = None
    data_source: Optional[str] = Field(default="Both", pattern="^(Agent Reports|Database Alerts|Both)$")


