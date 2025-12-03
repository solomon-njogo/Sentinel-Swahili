"""
Named Entity Recognition utilities for extracting Who, What, Where, When from Swahili threat reports.
Uses rule-based extraction first, then enhances with OpenRouter LLM for better accuracy.
"""

import re
import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import spacy
    from spacy.lang.en import English
    from spacy.lang.xx import MultiLanguage
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from agents.config import REQUIRED_FIELDS, MODEL_CONFIG
from agents.data_models import ExtractedEntities
from src.utils.logger import get_logger


class NERExtractor:
    """Extract named entities from Swahili threat reports with OpenRouter enhancement"""
    
    def __init__(self, model_name: Optional[str] = None, use_llm: bool = True):
        """
        Initialize NER extractor.
        
        Args:
            model_name: spaCy model name. If None, uses rule-based extraction optimized for Swahili.
            use_llm: Whether to use OpenRouter LLM for enhancement (default: True)
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.nlp = None
        self.use_llm = use_llm
        self.client = None
        
        if SPACY_AVAILABLE and model_name:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                # Model not found, will use rule-based extraction
                self.nlp = None
        
        # Initialize OpenRouter client for LLM enhancement
        if use_llm and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENROUTER_API_KEY") or MODEL_CONFIG.get("openrouter_api_key")
            base_url = MODEL_CONFIG.get("openrouter_base_url", "https://openrouter.ai/api/v1")
            model = MODEL_CONFIG.get("openrouter_model", "openai/gpt-oss-120b")
            
            if api_key:
                try:
                    self.client = openai.OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    self.llm_model = model
                    self.logger.info(f"Initialized OpenRouter client for NER enhancement with model: {model}")
                except Exception as e:
                    self.logger.warning(f"Could not initialize OpenRouter client: {e}. Using rule-based extraction only.")
                    self.client = None
            else:
                self.logger.debug("OpenRouter API key not found. Using rule-based extraction only.")
        else:
            if not OPENAI_AVAILABLE:
                self.logger.debug("OpenAI library not available. Using rule-based extraction only.")
    
    def extract_entities(self, text: str) -> ExtractedEntities:
        """
        Extract entities (Who, What, Where, When) from Swahili text.
        First uses rule-based extraction, then enhances with OpenRouter LLM if available.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            ExtractedEntities object with extracted information
        """
        # Step 1: Rule-based extraction (always done)
        entities = self._extract_with_rules_swahili(text)
        
        # Step 2: Enhance with OpenRouter LLM if available
        if self.client and self.use_llm:
            try:
                llm_entities = self._extract_with_llm(text)
                # Merge LLM results with rule-based results
                entities = self._merge_entities(entities, llm_entities)
                self.logger.debug("Enhanced entities with OpenRouter LLM")
            except Exception as e:
                self.logger.warning(f"Error enhancing entities with LLM: {e}. Using rule-based results only.")
        
        return entities
    
    def _extract_with_rules_swahili(self, text: str) -> ExtractedEntities:
        """Extract entities using rule-based patterns optimized for Swahili"""
        entities = ExtractedEntities()
        
        # Extract Who (Nani) - All people involved (victims, perpetrators, witnesses, etc.)
        # This includes first, second, and third person references
        
        # 1. Direct person references with roles/titles
        who_patterns = [
            # Person names with titles (perpetrators, victims, witnesses)
            r'\b(mwalimu|askari|polisi|raisi|bwana|bibi|daktari|profesa|mwanafunzi|mwanamke|mwanamume)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # Person names (capitalized, typically 2-3 words)
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        ]
        
        for pattern in who_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # Include both title and name, or just name
                        if len(match) > 1:
                            full_name = f"{match[0]} {match[1]}".strip()
                            entities.who.append(full_name)
                            entities.who.append(match[1].strip())  # Also add just the name
                        else:
                            if len(match[0].strip()) > 2:
                                entities.who.append(match[0].strip())
                    else:
                        if len(match.strip()) > 2:
                            entities.who.append(match.strip())
        
        # 2. Perpetrators/Attackers (washambuliaji, wafanya, etc.)
        perpetrator_patterns = [
            r'(?:washambuliaji|wafanya|waliofanya|wamefanya|waliohusika|wamehusika)\s*[:=]?\s*([^\.\n,]+)',
            r'(?:watu|mtu)\s+([^\.\n,]+?)\s+(?:wamevaa|amevaa|wamefanya|amefanya)',
            r'(?:watu|mtu)\s+([^\.\n,]+?)\s+(?:wamekuja|amekuja|wamefika|amefika)',
        ]
        
        for pattern in perpetrator_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        person = match[1] if len(match) > 1 else match[0]
                    else:
                        person = match
                    if len(person.strip()) > 2:
                        entities.who.append(person.strip())
        
        # 3. Victims (wamejeruhiwa, waliojeruhiwa, etc.)
        victim_patterns = [
            r'(?:wamejeruhiwa|waliojeruhiwa|wameuawa|walioauwa|wameathiriwa|walioathiriwa)\s*[:=]?\s*([^\.\n,]+)',
            r'(?:watu|mtu)\s+([^\.\n,]+?)\s+(?:wamejeruhiwa|amejeruhiwa|wameuawa|ameuawa)',
            r'(?:watu|mtu)\s+wengi\s+(?:wamejeruhiwa|wameuawa)',
        ]
        
        for pattern in victim_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        person = match[1] if len(match) > 1 else match[0]
                    else:
                        person = match
                    if len(person.strip()) > 2:
                        entities.who.append(person.strip())
        
        # 4. Witnesses/Observers (ameona, wameona, etc.)
        witness_patterns = [
            r'(?:ameona|wameona|aliona|waliona|ameona|wameona)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:ameona|aliona|wameona|waliona)',
        ]
        
        for pattern in witness_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        person = match[1] if len(match) > 1 else match[0]
                    else:
                        person = match
                    if len(person.strip()) > 2:
                        entities.who.append(person.strip())
        
        # 5. First person references (mimi, sisi - the reporter)
        first_person_patterns = [
            r'\b(mimi|sisi)\s+([^\.\n,]+?)(?:\.|,|na|kwa)',
            r'([^\.\n,]+?)\s+(?:nimeona|tumeona|nimefika|tumefika)',
        ]
        
        for pattern in first_person_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        person = match[1] if len(match) > 1 else "Reporter"
                    else:
                        person = "Reporter"
                    entities.who.append(person.strip())
        
        # 6. Second person references (wewe, nyinyi - if mentioned)
        second_person_patterns = [
            r'\b(wewe|nyinyi)\s+([^\.\n,]+?)(?:\.|,)',
        ]
        
        for pattern in second_person_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        person = match[1] if len(match) > 1 else match[0]
                    else:
                        person = match
                    if len(person.strip()) > 2:
                        entities.who.append(person.strip())
        
        # 7. Third person references (yeye, wao, etc.)
        third_person_patterns = [
            r'\b(yeye|wao)\s+([^\.\n,]+?)(?:\.|,|na)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:yeye|wao)\s+',
        ]
        
        for pattern in third_person_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        person = match[1] if len(match) > 1 else match[0]
                    else:
                        person = match
                    if len(person.strip()) > 2:
                        entities.who.append(person.strip())
        
        # 8. Generic person references with descriptions
        generic_patterns = [
            r'(?:watu|mtu)\s+([^\.\n,]+?)(?:\.|,|katika|kwa|wame|ame)',
            r'(?:watu|mtu)\s+wengi',
            r'(?:watu|mtu)\s+wa\s+([A-Z][a-z]+)',
        ]
        
        for pattern in generic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        person = match[1] if len(match) > 1 else match[0]
                    else:
                        person = match if match else "Watu wengi"
                    if len(person.strip()) > 2:
                        entities.who.append(person.strip())
        
        # Extract Where (Wapi) - Locations
        where_patterns = [
            # Major cities in East Africa
            r'\b(nairobi|mombasa|kisumu|dar es salaam|dar\s+es\s+salaam|arusha|dodoma|kampala|kigali|tanga|morogoro|zanzibar)\b',
            # "Wapi", "eneo", "mahali" indicators
            r'(?:wapi|eneo|mahali|katika|kwenye)\s*[:=]?\s*([^\.\n,]+?)(?:\.|,|saa|tarehe)',
            # Location patterns: "katika [location]" or "kwenye [location]"
            r'(?:katika|kwenye)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # Common location words
            r'\b(soko|shule|hospitali|stesheni|kituo|ofisi|jengo|nyumba)\s+(?:ya|la|cha)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # Street/road names
            r'\b(barabara|njia|street|road)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for pattern in where_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        location = match[1] if len(match) > 1 else match[0]
                        if len(location.strip()) > 2:
                            entities.where.append(location.strip())
                    else:
                        # Clean up common location words
                        location = match.strip()
                        if location and len(location) > 2:
                            # Remove common prefixes
                            location = re.sub(r'^(katika|kwenye|eneo|mahali)\s+', '', location, flags=re.IGNORECASE)
                            entities.where.append(location.strip())
        
        # Extract When (Lini) - Time/Date
        when_patterns = [
            # Swahili time words
            r'\b(leo|jana|kesho|sasa|hivi\s+sasa|usiku|mchana|asubuhi|jioni)\b',
            # Date patterns
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            # Time patterns
            r'\b(saa\s+\d{1,2}(?:\s*:\s*\d{2})?(?:\s*(?:asubuhi|mchana|jioni|usiku))?)\b',
            r'\b(\d{1,2}:\d{2}(?:\s*(?:am|pm|asubuhi|jioni|usiku))?)\b',
            # "Tarehe" or "Lini" indicators
            r'(?:tarehe|lini|wakati)\s*[:=]?\s*([^\.\n,]+?)(?:\.|,|katika)',
            # Day of week
            r'\b(jumatatu|jumanne|jumatano|alhamisi|ijumaa|jumamosi|jumapili)\b',
        ]
        
        for pattern in when_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        time_str = match[1] if len(match) > 1 else match[0]
                        if len(time_str.strip()) > 1:
                            entities.when.append(time_str.strip())
                    else:
                        if len(match.strip()) > 1:
                            entities.when.append(match.strip())
        
        # Extract What (Nini) - Threat/Event description
        what_keywords = self._extract_what_keywords_swahili(text)
        entities.what.extend(what_keywords)
        
        # Clean and deduplicate
        entities.who = self._clean_entities(list(dict.fromkeys(entities.who)))
        entities.where = self._clean_entities(list(dict.fromkeys(entities.where)))
        entities.when = self._clean_entities(list(dict.fromkeys(entities.when)))
        entities.what = self._clean_entities(list(dict.fromkeys(entities.what)))
        
        return entities
    
    def _extract_with_llm(self, text: str) -> ExtractedEntities:
        """
        Extract entities using OpenRouter LLM for enhanced accuracy.
        
        Args:
            text: Input text
            
        Returns:
            ExtractedEntities object with LLM-extracted entities
        """
        prompt = self._create_ner_prompt(text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in extracting named entities from Swahili threat reports. Extract Who, What, Where, and When information. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Low temperature for consistent extraction
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                # Create ExtractedEntities from LLM response
                entities = ExtractedEntities()
                entities.who = result.get("who", [])
                entities.what = result.get("what", [])
                entities.where = result.get("where", [])
                entities.when = result.get("when", [])
                
                # Clean entities
                entities.who = self._clean_entities(entities.who)
                entities.where = self._clean_entities(entities.where)
                entities.when = self._clean_entities(entities.when)
                entities.what = self._clean_entities(entities.what)
                
                return entities
            
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse LLM response as JSON, attempting text extraction")
                return self._extract_from_text_response(response_text)
        
        except Exception as e:
            self.logger.error(f"Error calling OpenRouter API for NER: {e}")
            return ExtractedEntities()  # Return empty entities on error
    
    def _create_ner_prompt(self, text: str) -> str:
        """Create prompt for LLM entity extraction"""
        return f"""Extract the following information from this Swahili threat report:

Threat Report:
{text}

Extract:
1. WHO (Nani): ALL people mentioned in the report, including:
   - Perpetrators/Attackers (washambuliaji, wafanya, etc.)
   - Victims (wamejeruhiwa, waliojeruhiwa, etc.)
   - Witnesses (ameona, wameona, etc.)
   - First person references (mimi, sisi - the reporter)
   - Second person references (wewe, nyinyi - if mentioned)
   - Third person references (yeye, wao, etc.)
   - People with titles/roles (mwalimu, askari, polisi, etc.)
   - Any person names mentioned (full names, first names, last names)
   - Generic person references (watu, mtu, etc.) with their descriptions

2. WHAT (Nini): The threat, event, or incident description
3. WHERE (Wapi): Locations, places, addresses, areas mentioned
4. WHEN (Lini): Time, date, when the event occurred or will occur

IMPORTANT for WHO: Extract ALL persons mentioned regardless of:
- Whether they are mentioned in first, second, or third person
- Whether they are victims, perpetrators, witnesses, or observers
- Whether they have full names, partial names, or just descriptions
- Whether they are mentioned explicitly or implicitly

Respond with a JSON object in this exact format:
{{
    "who": ["person1", "person2", "person3"],
    "what": ["threat description"],
    "where": ["location1", "location2"],
    "when": ["time/date1", "time/date2"]
}}

Only include entities that are clearly mentioned in the text. Be precise and comprehensive for WHO extraction."""
    
    def _extract_from_text_response(self, text: str) -> ExtractedEntities:
        """Extract entities from text response if JSON parsing fails"""
        entities = ExtractedEntities()
        text_lower = text.lower()
        
        # Try to extract entities from text format
        if "who" in text_lower or "nani" in text_lower:
            who_match = re.search(r'(?:who|nani)[:\s]+([^\n]+)', text, re.IGNORECASE)
            if who_match:
                entities.who = [w.strip() for w in who_match.group(1).split(",") if w.strip()]
        
        if "what" in text_lower or "nini" in text_lower:
            what_match = re.search(r'(?:what|nini)[:\s]+([^\n]+)', text, re.IGNORECASE)
            if what_match:
                entities.what = [w.strip() for w in what_match.group(1).split(",") if w.strip()]
        
        if "where" in text_lower or "wapi" in text_lower:
            where_match = re.search(r'(?:where|wapi)[:\s]+([^\n]+)', text, re.IGNORECASE)
            if where_match:
                entities.where = [w.strip() for w in where_match.group(1).split(",") if w.strip()]
        
        if "when" in text_lower or "lini" in text_lower:
            when_match = re.search(r'(?:when|lini)[:\s]+([^\n]+)', text, re.IGNORECASE)
            if when_match:
                entities.when = [w.strip() for w in when_match.group(1).split(",") if w.strip()]
        
        return entities
    
    def _merge_entities(self, base: ExtractedEntities, enhanced: ExtractedEntities) -> ExtractedEntities:
        """
        Merge rule-based entities with LLM-enhanced entities.
        LLM entities take priority, but rule-based entities are kept if they add value.
        
        Args:
            base: Rule-based extracted entities
            enhanced: LLM-extracted entities
            
        Returns:
            Merged ExtractedEntities
        """
        merged = ExtractedEntities()
        
        # For each field, combine and deduplicate
        # LLM results are prioritized, but we keep unique rule-based results
        
        # Who: Combine both, LLM first
        merged.who = list(dict.fromkeys(enhanced.who + base.who))
        
        # What: Combine both, prefer more specific descriptions
        merged.what = list(dict.fromkeys(enhanced.what + base.what))
        
        # Where: Combine both, LLM first
        merged.where = list(dict.fromkeys(enhanced.where + base.where))
        
        # When: Combine both, LLM first
        merged.when = list(dict.fromkeys(enhanced.when + base.when))
        
        # Clean merged entities
        merged.who = self._clean_entities(merged.who)
        merged.where = self._clean_entities(merged.where)
        merged.when = self._clean_entities(merged.when)
        merged.what = self._clean_entities(merged.what)
        
        return merged
    
    def _extract_what_keywords_swahili(self, text: str) -> List[str]:
        """Extract threat/event keywords (What) from Swahili text"""
        # Swahili threat keywords with context
        threat_patterns = [
            # Critical threats
            r'(shambulio|bomu|risasi|kuua|kushambulia|dharura|hatari\s+kubwa)',
            # High threats
            r'(silaha|tishio|hatari|kisu|bunduki)',
            # Medium threats
            r'(shughuli\s+za\s+kushuku|shughuli\s+isiyo\s+ya\s+kawaida|kushuku)',
            # General threats
            r'(wizi|ghasia|tishio|hatari)',
        ]
        
        found_events = []
        text_lower = text.lower()
        
        for pattern in threat_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract context around the keyword (5 words before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                # Clean up context
                context = re.sub(r'\s+', ' ', context)
                if len(context) > 10:  # Only add meaningful context
                    found_events.append(context)
                else:
                    found_events.append(match.group())
        
        # Also extract full sentences containing threat keywords
        sentences = re.split(r'[\.!?]\s+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in ['shambulio', 'bomu', 'risasi', 'silaha', 'tishio', 'hatari', 'shughuli za kushuku']):
                found_events.append(sentence.strip())
        
        return found_events[:5]  # Limit to 5 most relevant
    
    def _clean_entities(self, entities: List[str]) -> List[str]:
        """Clean and filter extracted entities"""
        cleaned = []
        for entity in entities:
            # Remove common stop words and prefixes
            cleaned_entity = re.sub(r'^(nani|wapi|lini|nini|katika|kwenye|eneo|mahali|tarehe|wakati|saa)\s+', '', entity, flags=re.IGNORECASE)
            cleaned_entity = cleaned_entity.strip()
            
            # Filter out very short or common words
            if len(cleaned_entity) > 2 and cleaned_entity.lower() not in ['na', 'ya', 'la', 'cha', 'wa', 'za']:
                cleaned.append(cleaned_entity)
        
        return cleaned


def calculate_field_completeness(entities: ExtractedEntities) -> Dict[str, float]:
    """
    Calculate completeness score for each field.
    
    Args:
        entities: Extracted entities
        
    Returns:
        Dictionary mapping field names to completeness scores (0.0-1.0)
    """
    scores = {}
    
    # Who: 1.0 if at least one person found, 0.0 otherwise
    scores["who"] = 1.0 if len(entities.who) > 0 else 0.0
    
    # What: 1.0 if at least one threat keyword found, 0.0 otherwise
    scores["what"] = 1.0 if len(entities.what) > 0 else 0.0
    
    # Where: 1.0 if at least one location found, 0.0 otherwise
    scores["where"] = 1.0 if len(entities.where) > 0 else 0.0
    
    # When: 1.0 if at least one time/date found, 0.0 otherwise
    scores["when"] = 1.0 if len(entities.when) > 0 else 0.0
    
    return scores
