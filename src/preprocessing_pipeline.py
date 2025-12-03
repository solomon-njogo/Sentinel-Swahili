"""
Preprocessing Pipeline for Swahili Dataset
Orchestrates reading, preprocessing, and writing of dataset files.
"""

import os
from pathlib import Path
from typing import Dict, List
from src.text_preprocessor import TextPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_file(input_path: str, output_path: str) -> Dict[str, int]:
    """
    Preprocess a single dataset file.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        
    Returns:
        Dictionary with statistics: {
            'total_lines': int,
            'processed_lines': int,
            'empty_lines': int,
            'unk_removed': int,
            'duplicates_removed': int
        }
    """
    logger.info(f"Processing file: {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    preprocessor = TextPreprocessor()
    statistics = {
        'total_lines': 0,
        'processed_lines': 0,
        'empty_lines': 0,
        'unk_removed': 0,
        'duplicates_removed': 0
    }
    
    # Read all lines
    lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            statistics['total_lines'] += 1
            original_line = line.rstrip('\n\r')
            lines.append(original_line)
    
    logger.info(f"Read {statistics['total_lines']} lines from {input_path}")
    
    # Count UNK tokens before preprocessing (count occurrences, not lines)
    unk_count_before = sum(line.upper().count('UNK') for line in lines)
    
    # Preprocess each line
    preprocessed_lines = []
    for line in lines:
        if not line.strip():
            statistics['empty_lines'] += 1
            preprocessed_lines.append(line)
            continue
        
        preprocessed = preprocessor.preprocess_line(line)
        
        if not preprocessed.strip():
            statistics['empty_lines'] += 1
            continue
        
        preprocessed_lines.append(preprocessed)
        statistics['processed_lines'] += 1
    
    # Count UNK tokens after preprocessing (count occurrences, not lines)
    unk_count_after = sum(line.upper().count('UNK') for line in preprocessed_lines)
    statistics['unk_removed'] = unk_count_before - unk_count_after
    
    # Remove duplicates
    deduplicated_lines, duplicates_count = preprocessor.remove_duplicates(preprocessed_lines)
    statistics['duplicates_removed'] = duplicates_count
    
    # Write preprocessed data to output file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in deduplicated_lines:
            f.write(line + '\n')
    
    final_line_count = len(deduplicated_lines)
    logger.info(f"Wrote {final_line_count} lines to {output_path}")
    
    return statistics


def preprocess_dataset(data_dir: str = "data") -> None:
    """
    Preprocess all dataset files in the specified directory.
    Processes train.txt, test.txt, and valid.txt if they exist.
    Saves preprocessed files to a 'preprocessed' subdirectory.
    
    Args:
        data_dir: Directory containing dataset files (default: "data")
    """
    logger.info(f"Starting dataset preprocessing in directory: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Create preprocessed output directory
    preprocessed_dir = data_path / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preprocessed files will be saved to: {preprocessed_dir}")
    
    # Dataset files to process
    dataset_files = ["train.txt", "test.txt", "valid.txt"]
    
    total_stats = {
        'total_lines': 0,
        'processed_lines': 0,
        'empty_lines': 0,
        'unk_removed': 0,
        'duplicates_removed': 0
    }
    
    for filename in dataset_files:
        input_path = data_path / filename
        
        if not input_path.exists():
            logger.warning(f"File not found, skipping: {input_path}")
            continue
        
        # Save to preprocessed folder with original filename
        output_path = preprocessed_dir / filename
        
        try:
            stats = preprocess_file(str(input_path), str(output_path))
            
            # Accumulate statistics
            for key in total_stats:
                total_stats[key] += stats[key]
            
            # Log file-specific statistics
            logger.info(f"Statistics for {filename}:")
            logger.info(f"  Total lines: {stats['total_lines']:,}")
            logger.info(f"  Processed lines: {stats['processed_lines']:,}")
            logger.info(f"  Empty lines: {stats['empty_lines']:,}")
            logger.info(f"  UNK tokens removed: {stats['unk_removed']:,}")
            logger.info(f"  Duplicates removed: {stats['duplicates_removed']:,}")
            logger.info(f"  Final line count: {stats['total_lines'] - stats['duplicates_removed']:,}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            raise
    
    # Log overall statistics
    logger.info("=" * 80)
    logger.info("Overall Preprocessing Statistics:")
    logger.info(f"  Total lines processed: {total_stats['total_lines']:,}")
    logger.info(f"  Successfully processed: {total_stats['processed_lines']:,}")
    logger.info(f"  Empty lines: {total_stats['empty_lines']:,}")
    logger.info(f"  Total UNK tokens removed: {total_stats['unk_removed']:,}")
    logger.info(f"  Total duplicates removed: {total_stats['duplicates_removed']:,}")
    logger.info("=" * 80)
    logger.info("Preprocessing complete!")

