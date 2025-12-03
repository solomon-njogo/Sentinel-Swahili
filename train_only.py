"""
Quick Training Script - Runs only the fine-tuning step
Skips preprocessing, EDA, and tokenization (assumes they're already done)
"""

from src.llama_finetuner import fine_tune_llama
from src.utils.logger import setup_logging_with_increment, get_logger
import logging

def main():
    """Run only the fine-tuning step."""
    # Setup logging
    log_file_path = setup_logging_with_increment(
        log_level=logging.INFO,
        log_dir="logs",
        prefix="training"
    )
    logger = get_logger(__name__)
    
    logger.info(f"Log file: {log_file_path}")
    logger.info("=" * 80)
    logger.info("Starting Llama 2 Fine-tuning (Training Only)")
    logger.info("=" * 80)
    
    try:
        # Run fine-tuning directly
        fine_tuning_results = fine_tune_llama(
            model_name="meta-llama/Llama-2-7b-hf",
            tokenized_dir="data/tokenized",
            output_dir="models/finetuned-llama2-7b",
            num_epochs=3,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_length=512,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            use_4bit=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            warmup_steps=100
        )
        
        if fine_tuning_results.get("success"):
            logger.info("=" * 80)
            logger.info("Training completed successfully!")
            logger.info("=" * 80)
        else:
            logger.error(f"Training failed: {fine_tuning_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

