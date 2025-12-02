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
    
    logger.info("Pipeline execution complete")


if __name__ == "__main__":
    main()
