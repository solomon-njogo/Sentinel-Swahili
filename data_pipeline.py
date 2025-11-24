"""
Data Pipeline Module for Swahili Text Processing
Handles data ingestion, parsing, and feature/target separation.
"""

import os
import re
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import logging
from text_cleaner import SwahiliTextCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Main class for data ingestion and parsing."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.label_detection_patterns = [
            r'^(\d+)\s+(.+)$',  # Numeric prefix
            r'^(.+)\t(.+)$',    # Tab-separated
            r'^(.+),(.+)$',     # Comma-separated
            r'^(.+)\|(.+)$',    # Pipe-separated
        ]
        # Initialize text cleaner for mandatory text cleaning
        self.text_cleaner = SwahiliTextCleaner()
        logger.info("DataPipeline initialized with text cleaning enabled")
    
    def read_file(self, filename: str) -> List[str]:
        """
        Read a text file line by line (memory-efficient for large files).
        
        Args:
            filename: Name of the file to read
            
        Returns:
            List of lines from the file
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Reading file: {filepath}")
        lines = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        lines.append(line)
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} lines...")
            
            logger.info(f"Successfully read {len(lines)} non-empty lines from {filename}")
            return lines
        
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise
    
    def detect_label_format(self, sample_lines: List[str]) -> Optional[str]:
        """
        Attempt to detect if labels are present and in what format.
        
        Args:
            sample_lines: Sample of lines to analyze
            
        Returns:
            Format type if detected, None otherwise
        """
        if not sample_lines:
            return None
        
        # Check a sample of lines
        sample_size = min(100, len(sample_lines))
        sample = sample_lines[:sample_size]
        
        for pattern_name, pattern in [
            ('numeric_prefix', r'^(\d+)\s+(.+)$'),
            ('tab_separated', r'^(.+)\t(.+)$'),
            ('comma_separated', r'^(.+),(.+)$'),
            ('pipe_separated', r'^(.+)\|(.+)$'),
        ]:
            matches = sum(1 for line in sample if re.match(pattern, line))
            if matches >= sample_size * 0.8:  # 80% match threshold
                logger.info(f"Detected label format: {pattern_name}")
                return pattern_name
        
        logger.info("No label format detected - assuming unlabeled data")
        return None
    
    def parse_line(self, line: str, label_format: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Parse a single line into feature (text) and target (label).
        
        Args:
            line: Raw line from file
            label_format: Detected label format or None
            
        Returns:
            Tuple of (text, label) where label may be None
        """
        if label_format is None:
            # No labels - entire line is the feature
            return (line, None)
        
        # Parse based on detected format
        if label_format == 'numeric_prefix':
            match = re.match(r'^(\d+)\s+(.+)$', line)
            if match:
                return (match.group(2), match.group(1))
        elif label_format == 'tab_separated':
            parts = line.split('\t', 1)
            if len(parts) == 2:
                return (parts[1], parts[0])
        elif label_format == 'comma_separated':
            parts = line.split(',', 1)
            if len(parts) == 2:
                return (parts[1], parts[0])
        elif label_format == 'pipe_separated':
            parts = line.split('|', 1)
            if len(parts) == 2:
                return (parts[1], parts[0])
        
        # Fallback: no label detected
        return (line, None)
    
    def parse_dataset(self, filename: str) -> Tuple[List[str], List[Optional[str]], Optional[str]]:
        """
        Parse a dataset file into features and targets.
        
        Args:
            filename: Name of the file to parse
            
        Returns:
            Tuple of (features, labels, detected_format)
        """
        logger.info(f"Parsing dataset: {filename}")
        
        # Read file
        lines = self.read_file(filename)
        
        if not lines:
            logger.warning(f"No data found in {filename}")
            return ([], [], None)
        
        # Detect label format
        label_format = self.detect_label_format(lines)
        
        # Parse lines and apply text cleaning
        features = []
        labels = []
        
        for line in lines:
            text, label = self.parse_line(line, label_format)
            # Apply mandatory text cleaning to the feature text
            cleaned_text = self.text_cleaner.clean(text)
            features.append(cleaned_text)
            labels.append(label)
        
        # Check if we actually found labels
        if label_format and all(l is None for l in labels):
            logger.warning(f"Label format detected but no labels extracted from {filename}")
            label_format = None
        
        labeled_count = sum(1 for l in labels if l is not None)
        logger.info(f"Parsed {len(features)} samples from {filename}")
        if labeled_count > 0:
            logger.info(f"Found {labeled_count} labeled samples")
        else:
            logger.info("No labels found - unlabeled dataset")
        
        return (features, labels, label_format)
    
    def get_dataset_info(self, filename: str) -> Dict[str, Any]:
        """
        Get basic information about a dataset file.
        
        Args:
            filename: Name of the file to analyze
            
        Returns:
            Dictionary with dataset information
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}
        
        # Count lines
        total_lines = 0
        empty_lines = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                if not line.strip():
                    empty_lines += 1
        
        return {
            "filename": filename,
            "total_lines": total_lines,
            "empty_lines": empty_lines,
            "non_empty_lines": total_lines - empty_lines,
            "file_size_mb": os.path.getsize(filepath) / (1024 * 1024)
        }


if __name__ == "__main__":
    # Test the pipeline
    pipeline = DataPipeline()
    
    # Test file reading
    for filename in ["train.txt", "test.txt", "valid.txt"]:
        info = pipeline.get_dataset_info(filename)
        print(f"\n{filename}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Parse dataset
        features, labels, format_type = pipeline.parse_dataset(filename)
        print(f"  Parsed samples: {len(features)}")
        print(f"  Label format: {format_type}")
        if labels and any(l is not None for l in labels):
            print(f"  Labeled samples: {sum(1 for l in labels if l is not None)}")
        
        # Show first few examples
        if features:
            print(f"\n  First example:")
            print(f"    Text: {features[0][:100]}...")
            if labels[0]:
                print(f"    Label: {labels[0]}")

