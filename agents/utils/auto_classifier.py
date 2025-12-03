"""
Auto-classification service to detect if WHERE/WHAT/WHO/WHEN information
is already present in the first message, allowing the system to skip unnecessary questions.
"""

import sys
from pathlib import Path
from typing import Dict, Set, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.utils.ner_extractor import NERExtractor
from agents.data_models import ExtractedEntities
from src.utils.logger import get_logger


class AutoClassifier:
    """Auto-classifies incoming messages to detect which fields are already provided"""
    
    def __init__(self):
        """Initialize auto-classifier with NER extractor"""
        self.logger = get_logger(__name__)
        self.ner_extractor = NERExtractor(use_llm=True)
    
    def classify_fields(self, message: str) -> Dict[str, bool]:
        """
        Classify which fields (WHERE, WHAT, WHO, WHEN) are present in the message.
        
        Args:
            message: The incoming message text
            
        Returns:
            Dictionary mapping field names to boolean indicating if field is present
            Example: {"where": True, "what": True, "who": False, "when": False}
        """
        self.logger.debug(f"Auto-classifying message: {message[:100]}...")
        
        # Extract entities using NER
        entities = self.ner_extractor.extract_entities(message)
        
        # Determine which fields are present
        # A field is considered present if it has at least one entity with reasonable confidence
        fields_present = {
            "where": len(entities.where) > 0 and any(len(loc.strip()) > 2 for loc in entities.where),
            "what": len(entities.what) > 0 and any(len(desc.strip()) > 3 for desc in entities.what),
            "who": len(entities.who) > 0 and any(len(person.strip()) > 2 for person in entities.who),
            "when": len(entities.when) > 0 and any(len(time.strip()) > 2 for time in entities.when)
        }
        
        self.logger.info(
            f"Auto-classification result: "
            f"WHERE={fields_present['where']}, "
            f"WHAT={fields_present['what']}, "
            f"WHO={fields_present['who']}, "
            f"WHEN={fields_present['when']}"
        )
        
        return fields_present
    
    def get_missing_fields(self, message: str, required_fields: list = None) -> list:
        """
        Get list of missing fields from the message.
        
        Args:
            message: The incoming message text
            required_fields: List of required field names. Defaults to ["where", "what", "who", "when"]
            
        Returns:
            List of field names that are missing from the message
        """
        if required_fields is None:
            required_fields = ["where", "what", "who", "when"]
        
        fields_present = self.classify_fields(message)
        missing = [field for field in required_fields if not fields_present.get(field, False)]
        
        return missing
    
    def get_next_question_field(self, message: str, question_order: list = None) -> Optional[str]:
        """
        Determine which field to ask about next based on what's missing.
        
        Args:
            message: The incoming message text
            question_order: Order in which to ask questions. Defaults to ["where", "what", "who", "when"]
            
        Returns:
            Name of the next field to ask about, or None if all fields are present
        """
        if question_order is None:
            question_order = ["where", "what", "who", "when"]
        
        fields_present = self.classify_fields(message)
        
        # Find first missing field in order
        for field in question_order:
            if not fields_present.get(field, False):
                return field
        
        # All fields present
        return None

