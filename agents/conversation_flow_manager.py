"""
Conversation Flow Manager for systematic threat reporting.
Handles step-by-step conversational flow asking WHERE, WHAT, WHO, WHEN questions one at a time.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.utils.auto_classifier import AutoClassifier
from agents.escalation_agent import EscalationAgent
from agents.data_models import SeverityLevel, ConversationState
from agents.config import CONVERSATION_FLOW_CONFIG, FLOW_MESSAGES
from src.utils.logger import get_logger


class ConversationFlowManager:
    """Manages the step-by-step conversation flow for threat reporting"""
    
    def __init__(self):
        """Initialize conversation flow manager"""
        self.logger = get_logger(__name__)
        self.auto_classifier = AutoClassifier()
        self.escalation_agent = EscalationAgent()
        self.config = CONVERSATION_FLOW_CONFIG
        self.question_order = self.config.get("question_order", ["where", "what", "who", "when"])
        self.messages = FLOW_MESSAGES
    
    def initialize_flow(self, first_message: str) -> Dict[str, Any]:
        """
        Initialize conversation flow from first message.
        Auto-classifies to detect what's already provided and checks for urgency.
        
        Args:
            first_message: The first message from the user
            
        Returns:
            Dictionary with flow state, collected data, and next action
        """
        self.logger.info("Initializing conversation flow")
        
        # Auto-classify to see what fields are already present
        fields_present = self.auto_classifier.classify_fields(first_message)
        
        # Check for urgency immediately
        urgency_result = self._check_urgency(first_message)
        
        # Determine which fields are missing
        missing_fields = [
            field for field in self.question_order
            if not fields_present.get(field, False)
        ]
        
        # Initialize collected data structure
        collected_data = {
            "where": None,
            "what": None,
            "who": None,
            "when": None
        }
        
        # Extract and store what's already provided
        entities = self.auto_classifier.ner_extractor.extract_entities(first_message)
        if fields_present["where"] and entities.where:
            collected_data["where"] = ", ".join(entities.where[:2])  # Take first 2 locations
        if fields_present["what"] and entities.what:
            collected_data["what"] = ", ".join(entities.what[:2])  # Take first 2 descriptions
        if fields_present["who"] and entities.who:
            collected_data["who"] = ", ".join(entities.who[:2])  # Take first 2 people
        if fields_present["when"] and entities.when:
            collected_data["when"] = ", ".join(entities.when[:2])  # Take first 2 time references
        
        # Determine next question
        next_field = missing_fields[0] if missing_fields else None
        current_state = self._field_to_state(next_field) if next_field else ConversationState.COMPLETE
        
        flow_state = {
            "state": current_state.value,
            "collected_data": collected_data,
            "missing_fields": missing_fields,
            "next_question_field": next_field,
            "is_urgent": urgency_result["is_urgent"],
            "urgency_severity": urgency_result["severity"].value if urgency_result["severity"] else None,
            "first_message": first_message,
            "started_at": datetime.now().isoformat()
        }
        
        self.logger.info(
            f"Flow initialized: state={current_state.value}, "
            f"missing={missing_fields}, urgent={urgency_result['is_urgent']}"
        )
        
        return flow_state
    
    def process_response(
        self,
        message: str,
        current_state: ConversationState,
        collected_data: Dict[str, Optional[str]]
    ) -> Dict[str, Any]:
        """
        Process user response and determine next action.
        
        Args:
            message: User's response message
            current_state: Current conversation state
            collected_data: Currently collected data
            
        Returns:
            Dictionary with updated flow state and next action
        """
        self.logger.info(f"Processing response in state: {current_state.value}")
        
        # Check if user wants to skip
        if self._is_skip_message(message):
            return self._handle_skip(current_state, collected_data)
        
        # Extract answer for current question
        current_field = self._state_to_field(current_state)
        if current_field:
            # Try to extract the answer from the message
            answer = self._extract_answer(message, current_field)
            collected_data[current_field] = answer if answer else message.strip()
        
        # Determine next missing field
        missing_fields = [
            field for field in self.question_order
            if collected_data.get(field) is None or collected_data.get(field).strip() == ""
        ]
        
        # Move to next question
        if missing_fields:
            next_field = missing_fields[0]
            next_state = self._field_to_state(next_field)
        else:
            # All fields collected
            next_state = ConversationState.COMPLETE
            next_field = None
        
        result = {
            "state": next_state.value,
            "collected_data": collected_data,
            "missing_fields": missing_fields,
            "next_question_field": next_field,
            "is_complete": next_state == ConversationState.COMPLETE,
            "needs_followup": self._needs_followup(message, current_field) if current_field else False
        }
        
        self.logger.info(
            f"Response processed: new_state={next_state.value}, "
            f"missing={missing_fields}, complete={result['is_complete']}"
        )
        
        return result
    
    def _check_urgency(self, message: str) -> Dict[str, Any]:
        """
        Check if message contains urgent/critical keywords.
        
        Args:
            message: Message to check
            
        Returns:
            Dictionary with urgency information
        """
        try:
            # Use escalation agent to check severity
            escalation_result = self.escalation_agent.escalate(text=message)
            
            is_urgent = (
                escalation_result.severity == SeverityLevel.CRITICAL or
                escalation_result.severity == SeverityLevel.HIGH
            )
            
            return {
                "is_urgent": is_urgent,
                "severity": escalation_result.severity,
                "urgency_keywords": escalation_result.urgency_keywords_found
            }
        except Exception as e:
            self.logger.warning(f"Error checking urgency: {e}")
            return {
                "is_urgent": False,
                "severity": None,
                "urgency_keywords": []
            }
    
    def _is_skip_message(self, message: str) -> bool:
        """Check if message indicates user wants to skip"""
        skip_indicators = ["skip", "soma", "ruhusa", "haijalishi", "si muhimu"]
        message_lower = message.lower().strip()
        return message_lower in skip_indicators or message_lower.startswith("skip")
    
    def _handle_skip(
        self,
        current_state: ConversationState,
        collected_data: Dict[str, Optional[str]]
    ) -> Dict[str, Any]:
        """Handle skip response"""
        current_field = self._state_to_field(current_state)
        if current_field:
            collected_data[current_field] = "[SKIPPED]"
        
        # Move to next question
        missing_fields = [
            field for field in self.question_order
            if collected_data.get(field) is None or collected_data.get(field).strip() == ""
        ]
        
        if missing_fields:
            next_field = missing_fields[0]
            next_state = self._field_to_state(next_field)
        else:
            next_state = ConversationState.COMPLETE
            next_field = None
        
        return {
            "state": next_state.value,
            "collected_data": collected_data,
            "missing_fields": missing_fields,
            "next_question_field": next_field,
            "is_complete": next_state == ConversationState.COMPLETE,
            "was_skipped": True
        }
    
    def _extract_answer(self, message: str, field: str) -> Optional[str]:
        """Extract answer for specific field from message"""
        entities = self.auto_classifier.ner_extractor.extract_entities(message)
        
        if field == "where" and entities.where:
            return ", ".join(entities.where[:2])
        elif field == "what" and entities.what:
            return ", ".join(entities.what[:2])
        elif field == "who" and entities.who:
            return ", ".join(entities.who[:2])
        elif field == "when" and entities.when:
            return ", ".join(entities.when[:2])
        
        return None
    
    def _needs_followup(self, message: str, field: str) -> bool:
        """Check if answer is vague and needs follow-up"""
        # Very short answers might need follow-up
        if len(message.strip()) < 5:
            return True
        
        # Check for vague indicators
        vague_words = ["hapa", "hapo", "huko", "kitu", "mtu", "siku", "wakati"]
        message_lower = message.lower()
        vague_count = sum(1 for word in vague_words if word in message_lower)
        
        # If more than 2 vague words, might need follow-up
        return vague_count >= 2
    
    def _field_to_state(self, field: str) -> ConversationState:
        """Convert field name to conversation state"""
        mapping = {
            "where": ConversationState.ASKING_WHERE,
            "what": ConversationState.ASKING_WHAT,
            "who": ConversationState.ASKING_WHO,
            "when": ConversationState.ASKING_WHEN
        }
        return mapping.get(field, ConversationState.INITIAL)
    
    def _state_to_field(self, state: ConversationState) -> Optional[str]:
        """Convert conversation state to field name"""
        mapping = {
            ConversationState.ASKING_WHERE: "where",
            ConversationState.ASKING_WHAT: "what",
            ConversationState.ASKING_WHO: "who",
            ConversationState.ASKING_WHEN: "when"
        }
        return mapping.get(state)
    
    def get_question_message(self, field: str, is_followup: bool = False) -> str:
        """
        Get the question message for a specific field.
        
        Args:
            field: Field name (where, what, who, when)
            is_followup: Whether this is a follow-up question
            
        Returns:
            Question message string
        """
        if is_followup:
            key = f"{field}_followup"
        else:
            key = f"{field}_question"
        
        return self.messages.get(key, self.messages.get(f"{field}_question", ""))
    
    def get_reassurance_message(self) -> str:
        """Get a reassurance message to show between questions"""
        return self.messages.get("reassurance", "Asante. Unaendelea vizuri. Swali moja tu zaidi...")
    
    def get_initial_greeting(self) -> str:
        """Get initial greeting message"""
        return self.messages.get("initial_greeting", 
            "Asante kwa kuripoti hii. Nitauliza maswali machache tu ili kuelewa hali ya hali zaidi.")

