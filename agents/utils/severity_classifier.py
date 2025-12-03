"""
Severity classification using OpenRouter LLM and urgency keyword detection.
Classifies threat reports into Low/Medium/High/Critical severity levels.
"""

from typing import List, Dict, Tuple, Optional
import re
import os
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from agents.config import ESCALATION_CONFIG, MODEL_CONFIG
from agents.data_models import SeverityLevel
from src.utils.logger import get_logger


class SeverityClassifier:
    """Classify threat severity using OpenRouter LLM and keyword detection"""
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize severity classifier.
        
        Args:
            model_name: OpenRouter model name. If None, uses config default.
            api_key: OpenRouter API key. If None, reads from environment or config.
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name or MODEL_CONFIG["openrouter_model"]
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or MODEL_CONFIG.get("openrouter_api_key")
        self.base_url = MODEL_CONFIG.get("openrouter_base_url", "https://openrouter.ai/api/v1")
        
        # Urgency keywords from config
        self.urgency_keywords = ESCALATION_CONFIG["urgency_keywords"]
        
        # Initialize OpenAI client if available
        self.client = None
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                self.logger.info(f"Initialized OpenRouter client with model: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenRouter client: {e}. Using keyword-based classification.")
                self.client = None
        else:
            if not OPENAI_AVAILABLE:
                self.logger.warning("OpenAI library not available. Using keyword-based classification.")
            if not self.api_key:
                self.logger.warning("OpenRouter API key not found. Using keyword-based classification.")
    
    def classify(self, text: str, use_keywords: bool = True) -> Tuple[SeverityLevel, float, List[str]]:
        """
        Classify threat severity.
        
        Args:
            text: Threat report text
            use_keywords: Whether to use keyword detection (default: True)
            
        Returns:
            Tuple of (severity_level, confidence_score, found_keywords)
        """
        found_keywords = []
        
        # First, check for urgency keywords
        if use_keywords:
            text_lower = text.lower()
            
            # Check critical keywords
            critical_keywords = [kw for kw in self.urgency_keywords["critical"] if kw.lower() in text_lower]
            if critical_keywords:
                found_keywords.extend(critical_keywords)
                # If critical keywords found, still use LLM but boost confidence
                if self.client:
                    llm_severity, llm_confidence, _ = self._classify_with_llm(text)
                    if llm_severity == SeverityLevel.CRITICAL or llm_severity == SeverityLevel.HIGH:
                        return SeverityLevel.CRITICAL, min(0.98, llm_confidence + 0.1), found_keywords
                    return SeverityLevel.CRITICAL, 0.95, found_keywords
                return SeverityLevel.CRITICAL, 0.95, found_keywords
            
            # Check high keywords
            high_keywords = [kw for kw in self.urgency_keywords["high"] if kw.lower() in text_lower]
            if high_keywords:
                found_keywords.extend(high_keywords)
            
            # Check medium keywords
            medium_keywords = [kw for kw in self.urgency_keywords["medium"] if kw.lower() in text_lower]
            if medium_keywords:
                found_keywords.extend(medium_keywords)
        
        # Use LLM if available, otherwise use keyword-based
        if self.client:
            return self._classify_with_llm(text, found_keywords)
        else:
            # Fallback to keyword-based classification
            if found_keywords:
                if any(kw in self.urgency_keywords["high"] for kw in found_keywords):
                    return SeverityLevel.HIGH, 0.85, found_keywords
                elif any(kw in self.urgency_keywords["medium"] for kw in found_keywords):
                    return SeverityLevel.MEDIUM, 0.70, found_keywords
            return SeverityLevel.LOW, 0.50, found_keywords
    
    def _classify_with_llm(self, text: str, found_keywords: List[str] = None) -> Tuple[SeverityLevel, float, List[str]]:
        """
        Classify using OpenRouter LLM.
        
        Args:
            text: Input text
            found_keywords: Pre-detected keywords (optional)
            
        Returns:
            Tuple of (severity_level, confidence_score, found_keywords)
        """
        if found_keywords is None:
            found_keywords = []
        
        # Create classification prompt
        prompt = self._create_classification_prompt(text)
        
        try:
            # Call OpenRouter API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a security threat analyst. Classify threat reports into severity levels: Low, Medium, High, or Critical. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=200
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                severity_str = result.get("severity", "low").upper()
                confidence = float(result.get("confidence", 0.7))
                reasoning = result.get("reasoning", "")
                
                # Map to SeverityLevel enum
                severity_map = {
                    "LOW": SeverityLevel.LOW,
                    "MEDIUM": SeverityLevel.MEDIUM,
                    "HIGH": SeverityLevel.HIGH,
                    "CRITICAL": SeverityLevel.CRITICAL,
                }
                
                severity = severity_map.get(severity_str, SeverityLevel.LOW)
                
                # Adjust confidence based on keywords
                if found_keywords:
                    if severity == SeverityLevel.CRITICAL and any(kw in self.urgency_keywords["critical"] for kw in found_keywords):
                        confidence = min(0.98, confidence + 0.1)
                    elif severity == SeverityLevel.HIGH and any(kw in self.urgency_keywords["high"] for kw in found_keywords):
                        confidence = min(0.95, confidence + 0.05)
                
                self.logger.debug(f"LLM classification: {severity.value} (confidence: {confidence:.2f})")
                return severity, confidence, found_keywords
            
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract severity from text
                self.logger.warning("Failed to parse LLM response as JSON, attempting text extraction")
                severity = self._extract_severity_from_text(response_text)
                confidence = 0.75  # Default confidence when parsing fails
                return severity, confidence, found_keywords
        
        except Exception as e:
            self.logger.error(f"Error calling OpenRouter API: {e}")
            # Fallback to keyword-based
            if found_keywords:
                if any(kw in self.urgency_keywords["critical"] for kw in found_keywords):
                    return SeverityLevel.CRITICAL, 0.90, found_keywords
                elif any(kw in self.urgency_keywords["high"] for kw in found_keywords):
                    return SeverityLevel.HIGH, 0.80, found_keywords
                elif any(kw in self.urgency_keywords["medium"] for kw in found_keywords):
                    return SeverityLevel.MEDIUM, 0.70, found_keywords
            return SeverityLevel.LOW, 0.50, found_keywords
    
    def _create_classification_prompt(self, text: str) -> str:
        """Create prompt for LLM classification"""
        return f"""Analyze the following threat report and classify its severity level.

Threat Report:
{text}

Classification Guidelines:
- CRITICAL: Immediate danger, active attacks, explosives, weapons in use, lives at risk
- HIGH: Weapons present, serious threats, potential for violence, requires urgent attention
- MEDIUM: Suspicious activity, concerning behavior, potential threats, needs monitoring
- LOW: Minor concerns, routine security matters, general observations

Respond with a JSON object in this exact format:
{{
    "severity": "CRITICAL|HIGH|MEDIUM|LOW",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""
    
    def _extract_severity_from_text(self, text: str) -> SeverityLevel:
        """Extract severity level from text response if JSON parsing fails"""
        text_upper = text.upper()
        
        if "CRITICAL" in text_upper:
            return SeverityLevel.CRITICAL
        elif "HIGH" in text_upper:
            return SeverityLevel.HIGH
        elif "MEDIUM" in text_upper:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def calculate_priority_score(self, severity: SeverityLevel, confidence: float, keyword_count: int) -> float:
        """
        Calculate priority score based on severity, confidence, and keywords.
        
        Args:
            severity: Severity level
            confidence: Classification confidence
            keyword_count: Number of urgency keywords found
            
        Returns:
            Priority score (0.0-1.0)
        """
        # Base priority by severity
        severity_weights = {
            SeverityLevel.LOW: 0.25,
            SeverityLevel.MEDIUM: 0.50,
            SeverityLevel.HIGH: 0.75,
            SeverityLevel.CRITICAL: 1.0,
        }
        
        base_priority = severity_weights.get(severity, 0.5)
        
        # Adjust by confidence
        adjusted_priority = base_priority * confidence
        
        # Boost by keyword count (up to 0.1)
        keyword_boost = min(0.1, keyword_count * 0.02)
        
        final_priority = min(1.0, adjusted_priority + keyword_boost)
        
        return final_priority
