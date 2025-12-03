"""
Text Preprocessor for Swahili Dataset
Handles UNK token removal, case normalization, punctuation normalization, and deduplication.
"""

import re
from typing import List, Tuple


class TextPreprocessor:
    """
    Preprocesses Swahili text by removing UNK tokens, normalizing case and punctuation,
    and removing duplicates.
    """
    
    # UNK token patterns to remove
    UNK_PATTERNS = [
        r'\bUNK\b',
        r'\bunk\b',
        r'<UNK>',
        r'\[UNK\]',
        r'<unk>',
        r'\[unk\]'
    ]
    
    def __init__(self):
        """Initialize the text preprocessor."""
        # Compile UNK regex pattern for efficiency
        unk_pattern = '|'.join(f'({pattern})' for pattern in self.UNK_PATTERNS)
        self.unk_regex = re.compile(unk_pattern, re.IGNORECASE)
    
    def remove_unk_tokens(self, text: str) -> str:
        """
        Remove all UNK token variants from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with UNK tokens removed
        """
        if not text:
            return ""
        
        # Remove all UNK token variants
        text = self.unk_regex.sub('', text)
        
        # Clean up extra spaces left by removed tokens
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def normalize_case(self, text: str) -> str:
        """
        Apply minimal case normalization while preserving most original casing.
        Only fixes obvious errors like ALL CAPS words in the middle of sentences.
        
        Args:
            text: Input text string
            
        Returns:
            Text with minimal case normalization applied
        """
        if not text:
            return ""
        
        # Split into words while preserving spaces
        words = text.split()
        normalized_words = []
        
        for i, word in enumerate(words):
            # Skip if word contains punctuation or is very short
            if len(word) <= 2 or not word.isalpha():
                normalized_words.append(word)
                continue
            
            # Check if word is ALL CAPS (potential error)
            if word.isupper() and len(word) > 2:
                # If it's the first word, keep it (might be intentional)
                # Otherwise, convert to title case
                if i == 0:
                    normalized_words.append(word)
                else:
                    # Check if previous word ended with sentence-ending punctuation
                    if i > 0 and normalized_words[-1] and normalized_words[-1][-1] in '.!?':
                        normalized_words.append(word)  # Keep caps after sentence end
                    else:
                        normalized_words.append(word.capitalize())
            else:
                # Preserve original casing
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize spacing around punctuation marks.
        Ensures single space after punctuation and removes multiple spaces.
        
        Args:
            text: Input text string
            
        Returns:
            Text with normalized punctuation spacing
        """
        if not text:
            return ""
        
        # Remove multiple consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure single space after punctuation marks
        # Pattern: punctuation followed by zero or more spaces, then a non-space character
        text = re.sub(r'([.,;:!?])\s*', r'\1 ', text)
        
        # Remove space before punctuation (if any)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Clean up any remaining multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def preprocess_line(self, text: str) -> str:
        """
        Apply all preprocessing steps to a single line of text.
        
        Processing order:
        1. Remove UNK tokens
        2. Normalize punctuation
        3. Normalize case (minimal)
        
        Args:
            text: Input text line
            
        Returns:
            Preprocessed text line
        """
        if not text or not text.strip():
            return ""
        
        # Step 1: Remove UNK tokens
        text = self.remove_unk_tokens(text)
        
        if not text.strip():
            return ""
        
        # Step 2: Normalize punctuation
        text = self.normalize_punctuation(text)
        
        if not text.strip():
            return ""
        
        # Step 3: Normalize case (minimal)
        text = self.normalize_case(text)
        
        return text.strip()
    
    def remove_duplicates(self, lines: List[str]) -> Tuple[List[str], int]:
        """
        Remove exact duplicate lines from a list.
        
        Args:
            lines: List of text lines
            
        Returns:
            Tuple of (deduplicated lines, count of duplicates removed)
        """
        seen = set()
        unique_lines = []
        duplicates_count = 0
        
        for line in lines:
            # Normalize line for comparison (strip whitespace)
            normalized = line.strip()
            
            # Skip empty lines in duplicate checking
            if not normalized:
                unique_lines.append(line)
                continue
            
            # Check if we've seen this line before
            if normalized in seen:
                duplicates_count += 1
            else:
                seen.add(normalized)
                unique_lines.append(line)
        
        return unique_lines, duplicates_count

