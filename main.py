"""
Main Pipeline for Swahili Text Processing and Transformer Fine-tuning
Orchestrates the complete pipeline from data loading to model training.

Usage:
    # Data preparation only:
    python main.py
   
    # Logger example:
    python main.py
"""

import logging
from src.utils.logger import setup_logging, get_logger, TRACE_LEVEL


def main():
    """Main execution function - orchestrates the complete pipeline."""
    # Setup logging with Loguru-style formatting
    # This will create colored output in terminal and plain text in log file
    setup_logging(
        log_level=TRACE_LEVEL,  # Set to TRACE_LEVEL to see all levels including TRACE
        log_file="example.log",
        log_dir="logs",
        use_colors=True
    )
    
    # Get logger for this module
    logger = get_logger(__name__)
    
    print("\n" + "="*80)
    print("  Logger Example - Loguru Style Formatting")
    print("="*80 + "\n")
    
    # Demonstrate all log levels
    logger.trace("A trace message.")
    logger.debug("A debug message.")
    logger.info("An info message.")
    logger.success("A success message.")
    logger.warning("A warning message.")
    logger.error("An error message.")
    


if __name__ == "__main__":
    main()
