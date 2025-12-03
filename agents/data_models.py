"""
Data models for threat report processing.
Defines structured data classes for reports, validation, and escalation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class SeverityLevel(str, Enum):
    """Threat severity levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ValidationStatus(str, Enum):
    """Validation status"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class ConversationState(str, Enum):
    """Conversation flow state"""
    INITIAL = "initial"
    ASKING_WHERE = "asking_where"
    ASKING_WHAT = "asking_what"
    ASKING_WHO = "asking_who"
    ASKING_WHEN = "asking_when"
    COMPLETE = "complete"


@dataclass
class ExtractedEntities:
    """Extracted named entities from a threat report"""
    who: List[str] = field(default_factory=list)  # Person/People involved
    what: List[str] = field(default_factory=list)  # Event/Threat description
    where: List[str] = field(default_factory=list)  # Location(s)
    when: List[str] = field(default_factory=list)  # Time/Date references
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FieldCompleteness:
    """Completeness score for a required field"""
    field_name: str
    score: float  # 0.0 to 1.0
    is_present: bool
    extracted_value: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ValidationResult:
    """Result from Validator Agent"""
    status: ValidationStatus
    overall_completeness: float  # 0.0 to 1.0
    entities: ExtractedEntities
    field_scores: List[FieldCompleteness]
    missing_fields: List[str]
    prompts: List[str]  # Prompts for missing information
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable datetime"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['status'] = self.status.value
        return result


@dataclass
class EscalationResult:
    """Result from Escalation Agent"""
    severity: SeverityLevel
    priority_score: float  # 0.0 to 1.0
    escalation_window_minutes: int
    urgency_keywords_found: List[str]
    classification_confidence: float
    requires_immediate_alert: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable datetime"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['severity'] = self.severity.value
        return result


@dataclass
class ThreatReport:
    """Complete threat report structure"""
    report_id: str
    raw_message: str
    source: str = "whatsapp"  # Source of the report
    received_at: datetime = field(default_factory=datetime.now)
    validation_result: Optional[ValidationResult] = None
    escalation_result: Optional[EscalationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable datetime"""
        result = asdict(self)
        result['received_at'] = self.received_at.isoformat()
        if self.validation_result:
            result['validation_result'] = self.validation_result.to_dict()
        if self.escalation_result:
            result['escalation_result'] = self.escalation_result.to_dict()
        return result


@dataclass
class ProcessedReport:
    """Final processed report ready for storage and dashboard"""
    report_id: str
    raw_message: str
    source: str
    received_at: datetime
    validation: ValidationResult
    escalation: EscalationResult
    status: str = "processed"
    processed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Structured collected data from conversation flow
    collected_where: Optional[str] = None
    collected_what: Optional[str] = None
    collected_who: Optional[str] = None
    collected_when: Optional[str] = None
    conversation_flow: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'report_id': self.report_id,
            'raw_message': self.raw_message,
            'source': self.source,
            'received_at': self.received_at.isoformat(),
            'processed_at': self.processed_at.isoformat(),
            'status': self.status,
            'validation': self.validation.to_dict(),
            'escalation': self.escalation.to_dict(),
            'metadata': self.metadata
        }
        # Add structured collected data if present
        if self.collected_where is not None:
            result['collected_where'] = self.collected_where
        if self.collected_what is not None:
            result['collected_what'] = self.collected_what
        if self.collected_who is not None:
            result['collected_who'] = self.collected_who
        if self.collected_when is not None:
            result['collected_when'] = self.collected_when
        if self.conversation_flow:
            result['conversation_flow'] = self.conversation_flow
        return result

