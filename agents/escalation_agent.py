"""
Escalation Agent: Classifies threat severity and assigns priority.
Makes autonomous escalation decisions based on severity levels.
"""

from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data_models import (
    EscalationResult,
    SeverityLevel,
    ValidationResult
)
from agents.config import ESCALATION_CONFIG
from agents.utils.severity_classifier import SeverityClassifier
from src.utils.logger import get_logger


class EscalationAgent:
    """Autonomous agent for threat severity classification and escalation"""
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Escalation Agent.
        
        Args:
            model_name: OpenRouter model name for classification. If None, uses config default.
            api_key: OpenRouter API key. If None, reads from environment or config.
        """
        self.logger = get_logger(__name__)
        self.classifier = SeverityClassifier(model_name=model_name, api_key=api_key)
        self.config = ESCALATION_CONFIG
    
    def escalate(
        self,
        text: str,
        validation_result: Optional[ValidationResult] = None
    ) -> EscalationResult:
        """
        Classify threat severity and determine escalation.
        
        Args:
            text: Threat report text
            validation_result: Optional validation result for context
            
        Returns:
            EscalationResult with severity, priority, and escalation window
        """
        self.logger.info("Starting escalation classification")

        # Build structured context from validation result when available.
        context = None
        if validation_result is not None:
            try:
                entities = validation_result.entities
                context = {
                    "raw_text": text,
                    "who": getattr(entities, "who", []) or [],
                    "what": getattr(entities, "what", []) or [],
                    "where": getattr(entities, "where", []) or [],
                    "when": getattr(entities, "when", []) or [],
                    "missing_fields": validation_result.missing_fields,
                    "overall_completeness": validation_result.overall_completeness,
                    "status": validation_result.status.value,
                }
            except Exception as e:
                # Context is helpful but not critical; log and continue.
                self.logger.warning(f"Error building escalation context: {e}")
                context = None
        
        # Classify severity using text plus optional structured context
        severity, confidence, urgency_keywords = self.classifier.classify(
            text,
            use_keywords=True,
            context=context
        )
        
        self.logger.info(
            f"Classified severity: {severity.value}, "
            f"confidence: {confidence:.2f}, "
            f"keywords found: {len(urgency_keywords)}"
        )
        
        # Calculate priority score
        priority_score = self.classifier.calculate_priority_score(
            severity=severity,
            confidence=confidence,
            keyword_count=len(urgency_keywords)
        )
        
        # Determine escalation window
        escalation_window = self._get_escalation_window(severity)
        
        # Check if immediate alert is required
        requires_immediate_alert = severity == SeverityLevel.CRITICAL
        
        # Create escalation result
        result = EscalationResult(
            severity=severity,
            priority_score=priority_score,
            escalation_window_minutes=escalation_window,
            urgency_keywords_found=urgency_keywords,
            classification_confidence=confidence,
            requires_immediate_alert=requires_immediate_alert
        )
        
        self.logger.info(
            f"Escalation complete: {severity.value}, "
            f"priority: {priority_score:.2f}, "
            f"window: {escalation_window} minutes"
        )
        
        return result
    
    def _get_escalation_window(self, severity: SeverityLevel) -> int:
        """
        Get escalation window in minutes based on severity.
        
        Args:
            severity: Severity level
            
        Returns:
            Escalation window in minutes
        """
        windows = self.config["escalation_windows"]
        
        severity_map = {
            SeverityLevel.CRITICAL: windows["critical"],
            SeverityLevel.HIGH: windows["high"],
            SeverityLevel.MEDIUM: windows["medium"],
            SeverityLevel.LOW: windows["low"],
        }
        
        return severity_map.get(severity, windows["low"])
    
    def should_escalate_immediately(self, severity: SeverityLevel) -> bool:
        """
        Determine if threat should be escalated immediately.
        
        Args:
            severity: Severity level
            
        Returns:
            True if immediate escalation is required
        """
        return severity == SeverityLevel.CRITICAL
    
    def get_escalation_action(self, severity: SeverityLevel) -> str:
        """
        Get recommended escalation action based on severity.
        
        Args:
            severity: Severity level
            
        Returns:
            Escalation action description
        """
        actions = {
            SeverityLevel.CRITICAL: "Immediate SMS alert to security dashboard (<2 min)",
            SeverityLevel.HIGH: "30-minute escalation window - notify security team",
            SeverityLevel.MEDIUM: "Daily intelligence digest",
            SeverityLevel.LOW: "Weekly intelligence digest"
        }
        
        return actions.get(severity, "Weekly intelligence digest")

