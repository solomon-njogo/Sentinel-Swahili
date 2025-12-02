"""
Configuration settings for the multi-agent threat processing system.
"""

from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # NER Model - using spaCy multilingual model
    "ner_model": "xx_ent_wiki_sm",  # Multilingual spaCy model
    "ner_model_path": None,  # Will download if not found
    
    # OpenRouter LLM for severity classification
    "openrouter_api_key": None,  # Set via environment variable OPENROUTER_API_KEY
    "openrouter_model": "openai/gpt-oss-120b",  # OpenRouter model for classification
    "openrouter_base_url": "https://openrouter.ai/api/v1",
    
    # Sentence transformer for semantic similarity
    "sentence_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

# Validation thresholds
VALIDATION_CONFIG = {
    "completeness_threshold": 0.7,  # Minimum completeness score to pass
    "field_weights": {
        "who": 0.25,
        "what": 0.35,  # Most important
        "where": 0.25,
        "when": 0.15,
    },
    "min_confidence": 0.5,  # Minimum confidence for entity extraction
}

# Escalation configuration
ESCALATION_CONFIG = {
    "severity_thresholds": {
        "critical": 0.85,
        "high": 0.70,
        "medium": 0.50,
        "low": 0.0,
    },
    "escalation_windows": {
        "critical": 2,  # minutes
        "high": 30,  # minutes
        "medium": 1440,  # 24 hours (daily digest)
        "low": 10080,  # 7 days (weekly digest)
    },
    "urgency_keywords": {
        "critical": [
            # English
            "attack", "bomb", "explosive", "gun", "shooting", "hostage",
            "terrorist", "immediate", "urgent", "now", "emergency",
            # Swahili
            "shambulio", "bomu", "risasi", "kushambulia", "dharura",
            "haraka", "sasa", "hatari", "tishio", "kuua"
        ],
        "high": [
            # English
            "weapon", "threat", "suspicious", "danger", "risk",
            # Swahili
            "silaha", "tishio", "shaka", "hatari", "hatari kubwa"
        ],
        "medium": [
            # English
            "suspicious activity", "unusual", "concerning",
            # Swahili
            "shughuli za kushuku", "isiyo ya kawaida", "inayosumbua"
        ],
    },
    "classification_confidence_threshold": 0.6,
}

# Storage configuration
STORAGE_CONFIG = {
    "storage_dir": STORAGE_DIR,
    "file_format": "json",
    "backup_enabled": False,
    "max_files": 10000,  # Maximum number of stored reports
}

# Logging configuration
LOGGING_CONFIG = {
    "log_dir": "logs",
    "log_prefix": "agents",
    "log_level": "INFO",
}

# Required fields for validation
REQUIRED_FIELDS = ["who", "what", "where", "when"]

# Field prompts (Swahili and English)
FIELD_PROMPTS = {
    "who": {
        "sw": "Tafadhali toa taarifa za watu waliohusika (Nani?)",
        "en": "Please provide information about the people involved (Who?)"
    },
    "what": {
        "sw": "Tafadhali eleza kile kilichotokea au tishio (Nini?)",
        "en": "Please describe what happened or the threat (What?)"
    },
    "where": {
        "sw": "Tafadhali toa eneo au mahali (Wapi?)",
        "en": "Please provide the location (Where?)"
    },
    "when": {
        "sw": "Tafadhali toa wakati au tarehe (Lini?)",
        "en": "Please provide the time or date (When?)"
    }
}

# Language detection
LANGUAGE_CONFIG = {
    "primary_language": "sw",  # Swahili
    "fallback_language": "en",  # English
    "supported_languages": ["sw", "en"],
}

