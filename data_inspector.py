"""
Data Inspector Module for Swahili Text Analysis
Handles data quality checks, statistics, and UNK token analysis.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataInspector:
    """Class for analyzing data quality and statistics."""
    
    def __init__(self):
        """Initialize the data inspector."""
        self.unk_token = "UNK"
    
    def count_tokens(self, text: str) -> Tuple[int, int]:
        """
        Count total tokens and UNK tokens in text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (total_tokens, unk_count)
        """
        tokens = text.split()
        total = len(tokens)
        unk_count = sum(1 for token in tokens if token == self.unk_token)
        return (total, unk_count)
    
    def analyze_text_length(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze text length statistics.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with length statistics
        """
        if not texts:
            return {}
        
        char_lengths = [len(text) for text in texts]
        word_lengths = [len(text.split()) for text in texts]
        
        return {
            "char_length": {
                "mean": sum(char_lengths) / len(char_lengths),
                "min": min(char_lengths),
                "max": max(char_lengths),
                "median": sorted(char_lengths)[len(char_lengths) // 2]
            },
            "word_length": {
                "mean": sum(word_lengths) / len(word_lengths),
                "min": min(word_lengths),
                "max": max(word_lengths),
                "median": sorted(word_lengths)[len(word_lengths) // 2]
            },
            "total_samples": len(texts)
        }
    
    def analyze_unk_tokens(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze UNK token frequency and patterns.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with UNK token statistics
        """
        if not texts:
            return {}
        
        total_tokens = 0
        total_unk = 0
        samples_with_unk = 0
        unk_positions = []  # Track position of UNK in sentences
        unk_contexts = []  # Track context around UNK tokens
        
        for text in texts:
            tokens = text.split()
            total_tokens += len(tokens)
            
            text_unk_count = 0
            for i, token in enumerate(tokens):
                if token == self.unk_token:
                    total_unk += 1
                    text_unk_count += 1
                    
                    # Track position (normalized: 0.0 = start, 1.0 = end)
                    position = i / len(tokens) if len(tokens) > 0 else 0.0
                    unk_positions.append(position)
                    
                    # Track context (previous and next token)
                    context = {
                        "prev": tokens[i-1] if i > 0 else "<START>",
                        "next": tokens[i+1] if i < len(tokens)-1 else "<END>"
                    }
                    unk_contexts.append(context)
            
            if text_unk_count > 0:
                samples_with_unk += 1
        
        # Calculate position statistics
        position_stats = {}
        if unk_positions:
            position_stats = {
                "mean": sum(unk_positions) / len(unk_positions),
                "min": min(unk_positions),
                "max": max(unk_positions),
                "median": sorted(unk_positions)[len(unk_positions) // 2]
            }
        
        # Analyze most common contexts
        prev_tokens = [ctx["prev"] for ctx in unk_contexts]
        next_tokens = [ctx["next"] for ctx in unk_contexts]
        most_common_prev = Counter(prev_tokens).most_common(10)
        most_common_next = Counter(next_tokens).most_common(10)
        
        return {
            "total_tokens": total_tokens,
            "total_unk_tokens": total_unk,
            "unk_percentage": (total_unk / total_tokens * 100) if total_tokens > 0 else 0.0,
            "samples_with_unk": samples_with_unk,
            "samples_without_unk": len(texts) - samples_with_unk,
            "unk_percentage_of_samples": (samples_with_unk / len(texts) * 100) if texts else 0.0,
            "position_statistics": position_stats,
            "most_common_prev_tokens": dict(most_common_prev),
            "most_common_next_tokens": dict(most_common_next),
            "unk_distribution": {
                "samples_with_0_unk": sum(1 for text in texts if text.split().count(self.unk_token) == 0),
                "samples_with_1_unk": sum(1 for text in texts if text.split().count(self.unk_token) == 1),
                "samples_with_2_unk": sum(1 for text in texts if text.split().count(self.unk_token) == 2),
                "samples_with_3plus_unk": sum(1 for text in texts if text.split().count(self.unk_token) >= 3),
            }
        }
    
    def analyze_labels(self, labels: List[Optional[str]]) -> Dict[str, Any]:
        """
        Analyze label distribution (if labels exist).
        
        Args:
            labels: List of labels (may contain None values)
            
        Returns:
            Dictionary with label statistics
        """
        if not labels:
            return {}
        
        labeled = [l for l in labels if l is not None]
        unlabeled = [l for l in labels if l is None]
        
        if not labeled:
            return {
                "total_samples": len(labels),
                "labeled_samples": 0,
                "unlabeled_samples": len(unlabeled),
                "label_distribution": {}
            }
        
        label_counts = Counter(labeled)
        
        return {
            "total_samples": len(labels),
            "labeled_samples": len(labeled),
            "unlabeled_samples": len(unlabeled),
            "num_unique_labels": len(label_counts),
            "label_distribution": dict(label_counts),
            "most_common_labels": dict(label_counts.most_common(10)),
            "class_balance": {
                "min_count": min(label_counts.values()),
                "max_count": max(label_counts.values()),
                "mean_count": sum(label_counts.values()) / len(label_counts)
            }
        }
    
    def check_data_quality(self, texts: List[str]) -> Dict[str, Any]:
        """
        Perform general data quality checks.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with quality metrics
        """
        if not texts:
            return {}
        
        empty_texts = sum(1 for text in texts if not text.strip())
        whitespace_only = sum(1 for text in texts if text.strip() == "")
        
        # Check for encoding issues (non-printable characters)
        encoding_issues = 0
        for text in texts:
            try:
                text.encode('utf-8').decode('utf-8')
            except:
                encoding_issues += 1
        
        # Check for special characters
        special_chars = Counter()
        for text in texts:
            for char in text:
                if not char.isalnum() and char not in ' \t\n\r':
                    special_chars[char] += 1
        
        return {
            "empty_texts": empty_texts,
            "whitespace_only": whitespace_only,
            "encoding_issues": encoding_issues,
            "special_characters": dict(special_chars.most_common(20)),
            "total_samples": len(texts)
        }
    
    def generate_unk_strategy(self, unk_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations for handling UNK tokens.
        
        Args:
            unk_stats: Statistics from analyze_unk_tokens
            
        Returns:
            Dictionary with strategy recommendations
        """
        if not unk_stats or unk_stats.get("total_tokens", 0) == 0:
            return {"strategy": "No UNK tokens found"}
        
        unk_percentage = unk_stats.get("unk_percentage", 0.0)
        samples_with_unk = unk_stats.get("samples_with_unk", 0)
        total_samples = unk_stats.get("samples_without_unk", 0) + samples_with_unk
        
        strategy = {
            "current_situation": {
                "unk_percentage_of_tokens": unk_percentage,
                "samples_affected": samples_with_unk,
                "percentage_of_samples": (samples_with_unk / total_samples * 100) if total_samples > 0 else 0.0
            },
            "recommended_strategy": "treat_as_unique_token",
            "rationale": "",
            "alternatives": []
        }
        
        if unk_percentage < 1.0:
            strategy["rationale"] = (
                f"UNK tokens represent only {unk_percentage:.2f}% of total tokens. "
                "Treating UNK as a unique vocabulary token is recommended as it preserves "
                "the original data structure and allows the model to learn UNK patterns."
            )
        elif unk_percentage < 5.0:
            strategy["rationale"] = (
                f"UNK tokens represent {unk_percentage:.2f}% of total tokens. "
                "While relatively low, treating UNK as a unique vocabulary token is still "
                "recommended. Consider investigating the source of UNK tokens for potential "
                "data preprocessing improvements."
            )
        else:
            strategy["rationale"] = (
                f"UNK tokens represent {unk_percentage:.2f}% of total tokens, which is significant. "
                "Treating UNK as a unique vocabulary token is recommended, but you may also consider: "
                "1) Investigating data preprocessing to reduce UNK generation, "
                "2) Using subword tokenization to handle rare words, "
                "3) Collecting more training data to cover vocabulary gaps."
            )
            strategy["alternatives"] = [
                "subword_tokenization",
                "data_preprocessing_review",
                "vocabulary_expansion"
            ]
        
        return strategy
    
    def inspect_dataset(self, features: List[str], labels: Optional[List[Optional[str]]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive inspection of a dataset.
        
        Args:
            features: List of text features
            labels: Optional list of labels
            
        Returns:
            Comprehensive statistics dictionary
        """
        logger.info(f"Inspecting dataset with {len(features)} samples...")
        
        results = {
            "text_statistics": self.analyze_text_length(features),
            "unk_statistics": self.analyze_unk_tokens(features),
            "quality_checks": self.check_data_quality(features),
        }
        
        if labels:
            results["label_statistics"] = self.analyze_labels(labels)
        
        # Generate UNK strategy
        results["unk_strategy"] = self.generate_unk_strategy(results["unk_statistics"])
        
        logger.info("Dataset inspection complete")
        return results
    
    def print_summary(self, stats: Dict[str, Any], dataset_name: str = "Dataset"):
        """
        Print a human-readable summary of statistics.
        
        Args:
            stats: Statistics dictionary from inspect_dataset
            dataset_name: Name of the dataset
        """
        print(f"\n{'='*60}")
        print(f"Summary for {dataset_name}")
        print(f"{'='*60}")
        
        # Text statistics
        if "text_statistics" in stats:
            ts = stats["text_statistics"]
            print(f"\nText Length Statistics:")
            print(f"  Total samples: {ts.get('total_samples', 0)}")
            if "char_length" in ts:
                cl = ts["char_length"]
                print(f"  Character length - Mean: {cl.get('mean', 0):.1f}, "
                      f"Min: {cl.get('min', 0)}, Max: {cl.get('max', 0)}, "
                      f"Median: {cl.get('median', 0)}")
            if "word_length" in ts:
                wl = ts["word_length"]
                print(f"  Word length - Mean: {wl.get('mean', 0):.1f}, "
                      f"Min: {wl.get('min', 0)}, Max: {wl.get('max', 0)}, "
                      f"Median: {wl.get('median', 0)}")
        
        # UNK statistics
        if "unk_statistics" in stats:
            us = stats["unk_statistics"]
            print(f"\nUNK Token Statistics:")
            print(f"  Total tokens: {us.get('total_tokens', 0):,}")
            print(f"  UNK tokens: {us.get('total_unk_tokens', 0):,}")
            print(f"  UNK percentage: {us.get('unk_percentage', 0):.2f}%")
            print(f"  Samples with UNK: {us.get('samples_with_unk', 0):,}")
            print(f"  Samples without UNK: {us.get('samples_without_unk', 0):,}")
            print(f"  Percentage of samples with UNK: {us.get('unk_percentage_of_samples', 0):.2f}%")
            
            if "unk_distribution" in us:
                dist = us["unk_distribution"]
                print(f"\n  UNK Distribution per Sample:")
                print(f"    0 UNK: {dist.get('samples_with_0_unk', 0):,}")
                print(f"    1 UNK: {dist.get('samples_with_1_unk', 0):,}")
                print(f"    2 UNK: {dist.get('samples_with_2_unk', 0):,}")
                print(f"    3+ UNK: {dist.get('samples_with_3plus_unk', 0):,}")
        
        # Label statistics
        if "label_statistics" in stats:
            ls = stats["label_statistics"]
            print(f"\nLabel Statistics:")
            print(f"  Total samples: {ls.get('total_samples', 0)}")
            print(f"  Labeled samples: {ls.get('labeled_samples', 0)}")
            print(f"  Unlabeled samples: {ls.get('unlabeled_samples', 0)}")
            if "num_unique_labels" in ls:
                print(f"  Unique labels: {ls.get('num_unique_labels', 0)}")
                if "most_common_labels" in ls:
                    print(f"  Most common labels:")
                    for label, count in list(ls["most_common_labels"].items())[:5]:
                        print(f"    {label}: {count}")
        
        # Quality checks
        if "quality_checks" in stats:
            qc = stats["quality_checks"]
            print(f"\nData Quality Checks:")
            print(f"  Empty texts: {qc.get('empty_texts', 0)}")
            print(f"  Encoding issues: {qc.get('encoding_issues', 0)}")
        
        # UNK Strategy
        if "unk_strategy" in stats:
            strategy = stats["unk_strategy"]
            print(f"\nUNK Token Strategy Recommendation:")
            print(f"  Strategy: {strategy.get('recommended_strategy', 'N/A')}")
            print(f"  Rationale: {strategy.get('rationale', 'N/A')}")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Import DataPipeline to load actual datasets
    from data_pipeline import DataPipeline
    
    # Initialize pipeline and inspector
    pipeline = DataPipeline(data_dir="data")
    inspector = DataInspector()
    
    # Process all datasets in the data folder
    datasets = ["train.txt", "test.txt", "valid.txt"]
    
    for filename in datasets:
        try:
            print(f"\n{'='*80}")
            print(f"Processing: {filename}")
            print(f"{'='*80}")
            
            # Parse dataset
            features, labels, label_format = pipeline.parse_dataset(filename)
            
            if not features:
                print(f"  WARNING: No data found in {filename}")
                continue
            
            # Inspect dataset
            stats = inspector.inspect_dataset(
                features, 
                labels if labels and any(l is not None for l in labels) else None
            )
            
            # Print summary
            dataset_name = filename.replace('.txt', '').upper()
            inspector.print_summary(stats, dataset_name)
        
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            import traceback
            traceback.print_exc()

