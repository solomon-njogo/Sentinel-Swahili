"""
Main Pipeline for Swahili Text Processing and Transformer Fine-tuning
Orchestrates the complete pipeline from data loading to model training.

Usage:
    # Run preprocessing pipeline:
    python main.py
"""

from src.preprocessing_pipeline import preprocess_dataset
from src.eda_analyzer import run_eda
from src.eda_visualizations import generate_all_visualizations
from src.llama_tokenizer import tokenize_dataset
from src.utils.logger import setup_logging_with_increment, get_logger
import logging


def main():
    """Main execution function - orchestrates the complete pipeline."""
    # Setup logging with incrementing log file
    log_file_path = setup_logging_with_increment(
        log_level=logging.INFO,
        log_dir="logs",
        prefix="pipeline"
    )
    logger = get_logger(__name__)
    
    logger.info(f"Log file: {log_file_path}")
    logger.info("Starting Swahili Dataset Preprocessing Pipeline")
    
    # Run preprocessing on dataset
    preprocess_dataset(data_dir="data")
    
    # Run Exploratory Data Analysis (EDA)
    logger.info("\n" + "=" * 80)
    logger.info("Starting Exploratory Data Analysis (EDA)")
    logger.info("=" * 80)
    
    try:
        # Run EDA analysis
        eda_results = run_eda(data_dir="data", output_dir="reports")
        
        # Generate visualizations
        generate_all_visualizations(eda_results, output_dir="reports")
        
        logger.info("EDA completed successfully!")
    except Exception as e:
        logger.error(f"Error during EDA: {e}", exc_info=True)
        logger.warning("Continuing despite EDA errors...")
    
    # Run Llama 2 Tokenization
    logger.info("\n" + "=" * 80)
    logger.info("Starting Llama 2 Tokenization")
    logger.info("=" * 80)
    
    try:
        tokenization_results = tokenize_dataset(
            data_dir="data",
            preprocessed_dir="data/preprocessed",
            output_dir="data/tokenized",
            block_size=512,
            stride=256
        )
        
        # Check if any tokenization succeeded
        if any(tokenization_results.values()):
            logger.info("Tokenization completed successfully!")
        else:
            logger.warning("Tokenization failed for all datasets. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Error during tokenization: {e}", exc_info=True)
        logger.warning("Continuing despite tokenization errors...")
    
    logger.info("Pipeline execution complete")


if __name__ == "__main__":
    main()
