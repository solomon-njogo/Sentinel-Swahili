"""
Configuration settings for the multi-agent threat processing system.
"""

import os
from pathlib import Path
from typing import Dict, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root (parent of agents directory)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, skip loading
    pass

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

# Twilio WhatsApp configuration
TWILIO_CONFIG = {
    "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
    "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
    "whatsapp_number": os.getenv("TWILIO_WHATSAPP_NUMBER"),  # Format: whatsapp:+14155238886
    "webhook_url": os.getenv("WEBHOOK_URL", "http://localhost:5000/webhook"),
    "port": int(os.getenv("PORT", "5000")),
}

# Conversation flow configuration
CONVERSATION_FLOW_CONFIG = {
    "session_timeout_seconds": 720,  # 12 minutes
    "question_order": ["where", "what", "who", "when"],
    "skip_keywords": ["skip", "soma", "ruhusa", "haijalishi", "si muhimu"],
    "vague_answer_threshold": 5,  # Minimum characters for non-vague answer
}

# Flow message templates (Swahili)
FLOW_MESSAGES = {
    "initial_greeting": (
        "Asante kwa kuripoti hii. Nitauliza maswali machache tu "
        "ili kuelewa hali ya hali zaidi."
    ),
    "where_question": (
        "Wapi hii inatokea? Eneo, jina la jengo, barabara, au eneo la jumla linatosha."
    ),
    "where_followup": (
        "Asante. Unaweza kutaja alama ya karibu au eneo? "
        "Unaweza pia kutuma eneo la moja kwa moja au pini kwenye ramani. "
        "Ikiwa hujui, jibu SKIP."
    ),
    "what_question": (
        "Ni nini hasa kinachotokea? "
        "(Mfano: mtu mwenye shaka, kitu cha hatari, tishio la mtandaoni, nk)"
    ),
    "what_followup": (
        "Nimeelewa. Unaweza kueleza kwa maneno machache zaidi? "
        "Ikiwa hujui, jibu SKIP."
    ),
    "who_question": (
        "Je, unajua nani anahusika? "
        "Jina, maelezo, au 'hajulikani' ni sawa."
    ),
    "who_followup": (
        "Maelezo yoyote madogo yanasaidia - mavazi, muonekano, au ukubwa wa kikundi. "
        "Au jibu SKIP."
    ),
    "when_question": (
        "Lini hii ilitokea au itatokea? "
        "(Wakati, tarehe, au 'sasa' ni sawa. Jibu SKIP ikiwa hujui.)"
    ),
    "when_followup": (
        "Unaweza kutaja wakati au tarehe maalum? "
        "Au jibu SKIP ikiwa hujui."
    ),
    "reassurance": (
        "Asante. Unaendelea vizuri. Swali moja tu zaidi..."
    ),
    "completion": (
        "Asante sana! Taarifa yako imekusanywa kikamilifu. "
        "Tutachambua na kuchukua hatua inayofaa."
    ),
}

# WhatsApp button configuration
BUTTON_CONFIG = {
    "skip_button_label": "SKIP",
    "skip_button_id": "skip",
}

