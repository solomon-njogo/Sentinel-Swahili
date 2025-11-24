"""
Text Cleaning Module for Swahili Text Analysis
Handles lowercase conversion, tokenization, stop-word removal, and stemming.
"""

import re
import logging
from typing import List, Set, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import NLTK
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    NLTK_AVAILABLE = True
    # Download required NLTK data if not already present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Using fallback implementations.")

# Comprehensive Swahili stop-word list
SWAHILI_STOP_WORDS = {
    # Common conjunctions and prepositions
    'na', 'ya', 'kwa', 'wa', 'za', 'la', 'katika', 'kama', 'hata', 'au', 'ama',
    # Demonstratives
    'hii', 'hiyo', 'hili', 'hivyo', 'hizi', 'hizo', 'huyu', 'huyo', 'hawa', 'hao',
    # Pronouns
    'ni', 'si', 'tu', 'wewe', 'yeye', 'sisi', 'nyinyi', 'wao',
    # Common verbs (auxiliary)
    'kuwa', 'kuwako', 'kuwepo',
    # Articles and particles
    'a', 'an', 'the',  # English articles that might appear
    # Common prepositions
    'kutoka', 'hadi', 'mpaka', 'kabla', 'baada', 'wakati', 'wakati wa',
    # Common adverbs
    'pia', 'sana', 'hapo', 'hapa', 'huko', 'pale',
    # Common question words
    'nani', 'nini', 'wapi', 'lini', 'kwa nini', 'vipi',
    # Common connectors
    'lakini', 'hata hivyo', 'kwa hiyo', 'kwa sababu', 'kwa kuwa',
    # Numbers (common ones)
    'moja', 'mbili', 'tatu', 'nne', 'tano',
    # Common time words
    'leo', 'jana', 'kesho', 'sasa', 'zamani',
}


class SwahiliStemmer:
    """
    Basic Swahili stemmer that handles common morphological patterns.
    """
    
    # Plural prefixes to remove
    PLURAL_PREFIXES = ['wa', 'vi', 'ma', 'mi', 'ny', 'n', 'u', 'ku', 'pa', 'mu']
    
    # Common suffixes to remove
    COMMON_SUFFIXES = ['ni', 'ka', 'ki', 'ko', 'ku', 'pa', 'po', 'mwa', 'mwenye']
    
    def stem(self, word: str) -> str:
        """
        Stem a Swahili word by removing plural prefixes and common suffixes.
        
        Args:
            word: Input word to stem
            
        Returns:
            Stemmed word
        """
        if not word or len(word) < 3:
            return word
        
        original_word = word
        
        # Remove plural prefixes (if word starts with them)
        for prefix in self.PLURAL_PREFIXES:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                # Check if removing prefix makes sense (word should be longer)
                potential_stem = word[len(prefix):]
                if len(potential_stem) >= 2:
                    word = potential_stem
                    break
        
        # Remove common suffixes (if word ends with them)
        for suffix in self.COMMON_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                # Check if removing suffix makes sense
                potential_stem = word[:-len(suffix)]
                if len(potential_stem) >= 2:
                    word = potential_stem
                    break
        
        # If stemming resulted in very short word, return original
        if len(word) < 2:
            return original_word
        
        return word


class SwahiliTextCleaner:
    """
    Class for cleaning and normalizing Swahili text.
    Performs: lowercase conversion, tokenization, stop-word removal, and stemming.
    """
    
    def __init__(self):
        """Initialize the text cleaner with stop-words and stemmer."""
        self.stop_words: Set[str] = self._load_stopwords()
        self.stemmer = self._load_stemmer()
        logger.info("SwahiliTextCleaner initialized")
    
    def _load_stopwords(self) -> Set[str]:
        """
        Load Swahili stop-words from available sources.
        
        Returns:
            Set of stop-words
        """
        stop_words = set()
        
        # Try NLTK first
        if NLTK_AVAILABLE:
            try:
                # NLTK doesn't have Swahili stopwords by default, but try anyway
                nltk_stopwords = set(stopwords.words('swahili'))
                if nltk_stopwords:
                    stop_words.update(nltk_stopwords)
                    logger.info("Loaded Swahili stop-words from NLTK")
            except (LookupError, OSError):
                # NLTK doesn't have Swahili stopwords
                pass
        
        # Always add our predefined list (as fallback or supplement)
        stop_words.update(SWAHILI_STOP_WORDS)
        
        if not stop_words:
            logger.warning("No stop-words loaded. Using empty set.")
        else:
            logger.info(f"Loaded {len(stop_words)} Swahili stop-words")
        
        return stop_words
    
    def _load_stemmer(self):
        """
        Load stemmer from available sources.
        
        Returns:
            Stemmer instance
        """
        # Try NLTK SnowballStemmer for Swahili
        if NLTK_AVAILABLE:
            try:
                # SnowballStemmer doesn't support Swahili, but try anyway
                stemmer = SnowballStemmer('swahili')
                logger.info("Loaded NLTK SnowballStemmer for Swahili")
                return stemmer
            except ValueError:
                # Swahili not supported by SnowballStemmer
                pass
        
        # Fallback to custom stemmer
        logger.info("Using custom Swahili stemmer")
        return SwahiliStemmer()
    
    def to_lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Input text string
            
        Returns:
            Lowercase text
        """
        return text.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        Handles punctuation and whitespace correctly.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words)
        """
        # Use regex to find word boundaries, preserving Swahili characters
        # This pattern matches word characters including Swahili-specific characters
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove Swahili stop-words from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stop-words removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of stemmed tokens
        """
        stemmed = []
        for token in tokens:
            if isinstance(self.stemmer, SwahiliStemmer):
                stemmed.append(self.stemmer.stem(token))
            else:
                # NLTK stemmer (if available)
                try:
                    stemmed.append(self.stemmer.stem(token))
                except:
                    # Fallback: return original token
                    stemmed.append(token)
        return stemmed
    
    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps to text in sequence:
        1. Convert to lowercase
        2. Tokenize
        3. Remove stop-words
        4. Stem
        5. Rejoin to string
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned and normalized text string
        """
        if not text or not text.strip():
            return ""
        
        # Step 1: Convert to lowercase
        text = self.to_lowercase(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(text)
        
        if not tokens:
            return ""
        
        # Step 3: Remove stop-words
        tokens = self.remove_stopwords(tokens)
        
        # Step 4: Stem
        tokens = self.stem(tokens)
        
        # Step 5: Rejoin tokens with spaces
        cleaned_text = " ".join(tokens)
        
        return cleaned_text


if __name__ == "__main__":
    # Test the text cleaner
    cleaner = SwahiliTextCleaner()
    
    # Test with sample Swahili text
    test_text = "Taarifa hiyo ilisema kuwa ongezeko la joto la maji juu ya wastani katikati ya bahari ya UNK inaashiria kuwepo kwa mvua za el nino"
    
    print("Original text:")
    print(test_text)
    print("\nCleaned text:")
    cleaned = cleaner.clean(test_text)
    print(cleaned)
    
    print("\nStep-by-step:")
    print(f"1. Lowercase: {cleaner.to_lowercase(test_text)}")
    tokens = cleaner.tokenize(test_text)
    print(f"2. Tokens: {tokens}")
    tokens_no_stop = cleaner.remove_stopwords(tokens)
    print(f"3. After stop-word removal: {tokens_no_stop}")
    stemmed = cleaner.stem(tokens_no_stop)
    print(f"4. After stemming: {stemmed}")

