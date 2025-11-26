"""
Text Cleaning Module for Swahili Text Analysis
Handles data cleaning: removes noise/boilerplate, ensures UTF-8 encoding,
removes tokenization placeholders (UNK), lowercase conversion, tokenization,
stop-word removal, and stemming.
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
    Performs: UTF-8 normalization, noise/boilerplate removal, UNK token removal,
    lowercase conversion, tokenization, stop-word removal, and stemming.
    """
    
    def __init__(self):
        """Initialize the text cleaner with stop-words and stemmer."""
        self.stop_words: Set[str] = self._load_stopwords()
        self.stemmer = self._load_stemmer()
        # Common tokenization placeholders to remove
        self.tokenization_placeholders = {'UNK', '<UNK>', '[UNK]', '<unk>', '[unk]'}
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
    
    def normalize_encoding(self, text: str) -> str:
        """
        Ensure consistent UTF-8 encoding by handling encoding errors.
        
        Args:
            text: Input text string
            
        Returns:
            UTF-8 normalized text
        """
        if not text:
            return ""
        
        # If text is already a string, ensure it's properly encoded
        try:
            # Encode to UTF-8 and decode back to ensure consistency
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            else:
                # Normalize by encoding and decoding
                text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.warning(f"Encoding error handled: {e}")
            # Replace problematic characters
            text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        return text
    
    def remove_noise(self, text: str) -> str:
        """
        Remove noise and boilerplate from text.
        Removes extra whitespace, leading/trailing spaces, and normalizes spacing.
        
        Args:
            text: Input text string
            
        Returns:
            Text with noise removed
        """
        if not text:
            return ""
        
        # Remove extra whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_tokenization_placeholders(self, tokens: List[str]) -> List[str]:
        """
        Remove tokenization placeholders like UNK from token list.
        Checks case-insensitively to catch UNK, unk, etc.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with placeholders removed
        """
        # Create a set of lowercase placeholders for case-insensitive matching
        placeholder_lower = {p.lower() for p in self.tokenization_placeholders}
        return [token for token in tokens if token.lower() not in placeholder_lower]
    
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
    
    def clean(self, text: str, remove_unk: bool = False) -> str:
        """
        Apply all cleaning steps to text in sequence:
        1. Normalize UTF-8 encoding
        2. Remove noise/boilerplate
        3. Convert to lowercase
        4. Tokenize
        5. Remove tokenization placeholders (UNK) - OPTIONAL
        6. Remove stop-words
        7. Stem
        8. Rejoin to string
        
        Args:
            text: Input text string
            remove_unk: If True, remove UNK tokens. If False, preserve them.
                       Default: False (preserve UNK for transformer compatibility)
        
        Returns:
            Cleaned and normalized text string
        """
        if not text or not text.strip():
            return ""
        
        # Step 1: Normalize UTF-8 encoding
        text = self.normalize_encoding(text)
        
        # Step 2: Remove noise/boilerplate
        text = self.remove_noise(text)
        
        if not text:
            return ""
        
        # Step 3: Convert to lowercase
        text = self.to_lowercase(text)
        
        # Step 4: Tokenize
        tokens = self.tokenize(text)
        
        if not tokens:
            return ""
        
        # Step 5: Remove tokenization placeholders (UNK, etc.) - OPTIONAL
        # For transformer models, UNK tokens should be preserved as they may
        # represent actual unknown words in the original data
        if remove_unk:
            tokens = self.remove_tokenization_placeholders(tokens)
        
        if not tokens:
            return ""
        
        # Step 6: Remove stop-words
        tokens = self.remove_stopwords(tokens)
        
        # Step 7: Stem
        tokens = self.stem(tokens)
        
        # Step 8: Rejoin tokens with spaces
        cleaned_text = " ".join(tokens)
        
        return cleaned_text


if __name__ == "__main__":
    # Process the entire dataset with text cleaner
    import os
    import sys
    from pathlib import Path
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_pipeline import DataPipeline
    
    cleaner = SwahiliTextCleaner()
    pipeline = DataPipeline(data_dir="data")
    
    # Process all dataset files
    dataset_files = ["train.txt", "test.txt", "valid.txt"]
    
    for filename in dataset_files:
        try:
            print(f"\n{'='*80}")
            print(f"Processing entire {filename} dataset")
            print(f"{'='*80}")
            
            # Parse entire dataset (this applies cleaning to all lines)
            features, labels, label_format = pipeline.parse_dataset(filename)
            
            if not features:
                print(f"  WARNING: No data found in {filename}")
                continue
            
            print(f"\n  Total samples processed: {len(features)}")
            print(f"  Label format: {label_format if label_format else 'Unlabeled'}")
            
            # Show statistics
            total_chars = sum(len(f) for f in features)
            total_words = sum(len(f.split()) for f in features)
            unk_count = sum(1 for f in features if 'unk' in f.lower())
            
            print(f"\n  Statistics:")
            print(f"    Total characters: {total_chars:,}")
            print(f"    Total words: {total_words:,}")
            print(f"    Average chars per sample: {total_chars // len(features):.1f}")
            print(f"    Average words per sample: {total_words // len(features):.1f}")
            print(f"    Samples with UNK (before cleaning): {unk_count}")
            
            # Show first example
            if features:
                print(f"\n  First example (cleaned):")
                print(f"    {features[0][:150]}...")
                
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Complete dataset processing finished!")
    print(f"{'='*80}\n")

