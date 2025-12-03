"""
Validator Agent: Ensures report completeness (Who? What? Where? When?)
Uses NER and semantic similarity scoring to validate threat reports.
"""

from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data_models import (
    ValidationResult,
    ValidationStatus,
    ExtractedEntities,
    FieldCompleteness
)
from agents.config import (
    VALIDATION_CONFIG,
    REQUIRED_FIELDS,
    FIELD_PROMPTS,
    LANGUAGE_CONFIG
)
from agents.utils.ner_extractor import NERExtractor, calculate_field_completeness
from src.utils.logger import get_logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class ValidatorAgent:
    """Autonomous agent for validating threat report completeness"""
    
    def __init__(self, ner_model: Optional[str] = None):
        """
        Initialize Validator Agent.
        
        Args:
            ner_model: spaCy model name for NER. If None, uses rule-based extraction.
        """
        self.logger = get_logger(__name__)
        # Initialize NER extractor with LLM enhancement enabled by default
        self.ner_extractor = NERExtractor(ner_model, use_llm=True)
        self.semantic_model = None
        
        # Load semantic similarity model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from agents.config import MODEL_CONFIG
                self.semantic_model = SentenceTransformer(MODEL_CONFIG["sentence_model"])
                self.logger.info("Loaded semantic similarity model")
            except Exception as e:
                self.logger.warning(f"Could not load semantic model: {e}. Using rule-based validation.")
    
    def validate(self, text: str, language: str = "sw") -> ValidationResult:
        """
        Validate threat report completeness (Swahili only).
        
        Args:
            text: Raw threat report text (must be in Swahili)
            language: Language code (default: "sw" for Swahili)
            
        Returns:
            ValidationResult with completeness scores and missing fields
        """
        """
        Validate threat report completeness.
        
        Args:
            text: Raw threat report text
            language: Language code ("sw" for Swahili, "en" for English)
            
        Returns:
            ValidationResult with completeness scores and missing fields
        """
        self.logger.info("Starting validation of threat report")
        
        # Extract entities
        entities = self.ner_extractor.extract_entities(text)
        self.logger.debug(f"Extracted entities: {entities.to_dict()}")
        
        # Calculate field completeness scores
        field_scores = calculate_field_completeness(entities)
        
        # Create FieldCompleteness objects
        field_completeness_list = []
        for field_name in REQUIRED_FIELDS:
            score = field_scores.get(field_name, 0.0)
            is_present = score > 0.5
            
            # Get extracted value
            extracted_value = None
            if field_name == "who" and entities.who:
                extracted_value = ", ".join(entities.who[:3])  # Limit to 3
            elif field_name == "what" and entities.what:
                extracted_value = ", ".join(entities.what[:3])
            elif field_name == "where" and entities.where:
                extracted_value = ", ".join(entities.where[:3])
            elif field_name == "when" and entities.when:
                extracted_value = ", ".join(entities.when[:3])
            
            # Calculate confidence (simple heuristic for now)
            confidence = score
            
            field_completeness = FieldCompleteness(
                field_name=field_name,
                score=score,
                is_present=is_present,
                extracted_value=extracted_value,
                confidence=confidence
            )
            field_completeness_list.append(field_completeness)
        
        # Calculate overall completeness using weighted average
        overall_completeness = self._calculate_overall_completeness(
            field_scores,
            VALIDATION_CONFIG["field_weights"]
        )
        
        # Determine missing fields
        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field_scores.get(field, 0.0) < VALIDATION_CONFIG["completeness_threshold"]
        ]
        
        # Determine validation status
        if overall_completeness >= VALIDATION_CONFIG["completeness_threshold"]:
            status = ValidationStatus.COMPLETE
        elif overall_completeness >= 0.4:
            status = ValidationStatus.INCOMPLETE
        else:
            status = ValidationStatus.FAILED
        
        # Generate prompts for missing fields
        prompts = self._generate_prompts(missing_fields, language)
        
        # Create validation result
        result = ValidationResult(
            status=status,
            overall_completeness=overall_completeness,
            entities=entities,
            field_scores=field_completeness_list,
            missing_fields=missing_fields,
            prompts=prompts
        )
        
        self.logger.info(
            f"Validation complete: {status.value}, "
            f"completeness: {overall_completeness:.2f}, "
            f"missing: {len(missing_fields)} fields"
        )
        
        return result
    
    def _calculate_overall_completeness(
        self,
        field_scores: dict,
        field_weights: dict
    ) -> float:
        """
        Calculate weighted overall completeness score.
        
        Args:
            field_scores: Dictionary of field completeness scores
            field_weights: Dictionary of field weights
            
        Returns:
            Overall completeness score (0.0-1.0)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for field_name, weight in field_weights.items():
            score = field_scores.get(field_name, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _generate_prompts(self, missing_fields: List[str], language: str) -> List[str]:
        """
        Generate prompts for missing fields.
        
        Args:
            missing_fields: List of missing field names
            language: Language code ("sw" or "en")
            
        Returns:
            List of prompt messages
        """
        prompts = []
        lang_key = language if language in ["sw", "en"] else LANGUAGE_CONFIG["fallback_language"]
        
        for field_name in missing_fields:
            if field_name in FIELD_PROMPTS:
                prompt = FIELD_PROMPTS[field_name].get(lang_key, FIELD_PROMPTS[field_name]["en"])
                prompts.append(prompt)
        
        return prompts
    
    def enhance_with_semantic_similarity(
        self,
        text: str,
        entities: ExtractedEntities
    ) -> ExtractedEntities:
        """
        Enhance entity extraction using semantic similarity.
        This is a placeholder for future enhancement.
        
        Args:
            text: Input text
            entities: Extracted entities
            
        Returns:
            Enhanced entities
        """
        if not self.semantic_model:
            return entities
        
        # Future: Use semantic similarity to find related entities
        # For now, return as-is
        return entities

