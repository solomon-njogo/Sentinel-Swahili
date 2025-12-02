"""
Main Pipeline for Swahili Text Processing and Transformer Fine-tuning
Orchestrates the complete pipeline from data loading to model training.

Usage:
    # Run preprocessing pipeline:
    python main.py
"""

from src.preprocessing_pipeline import preprocess_dataset
from src.utils.logger import setup_logging, get_logger
import logging


def main():
    """Main execution function - orchestrates the complete pipeline."""
    # Setup logging
    setup_logging(log_level=logging.INFO)
    logger = get_logger(__name__)
    
    logger.info("Starting Swahili Dataset Preprocessing Pipeline")
    
    # Run preprocessing on dataset
    preprocess_dataset(data_dir="data")
    
    logger.info("Pipeline execution complete")


if __name__ == "__main__":
    main()
