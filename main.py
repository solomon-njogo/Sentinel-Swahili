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
from src.llama_finetuner import fine_tune_llama
from src.model_evaluator import evaluate_model, generate_sample_predictions, save_evaluation_results
from src.training_visualizations import generate_all_training_visualizations
from src.utils.logger import setup_logging_with_increment, get_logger
from datasets import load_from_disk
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
    
    # Run Fine-tuning with QLoRA
    logger.info("\n" + "=" * 80)
    logger.info("Starting Llama 2 Fine-tuning with QLoRA")
    logger.info("=" * 80)
    
    try:
        # Check if tokenization succeeded before fine-tuning
        if any(tokenization_results.values()):
            # Run fine-tuning
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
                logger.info("Fine-tuning completed successfully!")
                
                # Generate training visualizations
                logger.info("\n" + "=" * 80)
                logger.info("Generating Training Visualizations")
                logger.info("=" * 80)
                
                try:
                    training_history = fine_tuning_results.get("training_history", [])
                    if training_history:
                        generate_all_training_visualizations(
                            training_history,
                            output_dir="reports"
                        )
                        logger.info("Training visualizations generated successfully!")
                    else:
                        logger.warning("No training history available for visualization")
                except Exception as e:
                    logger.error(f"Error generating visualizations: {e}", exc_info=True)
                
                # Evaluate model on test set
                logger.info("\n" + "=" * 80)
                logger.info("Evaluating Fine-tuned Model")
                logger.info("=" * 80)
                
                try:
                    # Load test dataset
                    test_dataset = None
                    test_path = "data/tokenized/test"
                    from pathlib import Path
                    if Path(test_path).exists():
                        test_dataset = load_from_disk(test_path)
                        logger.info(f"Loaded test dataset: {len(test_dataset):,} examples")
                    
                    # Evaluate
                    model_path = fine_tuning_results.get("output_dir", "models/finetuned-llama2-7b")
                    eval_results = evaluate_model(
                        model_path=model_path,
                        test_dataset=test_dataset,
                        max_length=512,
                        batch_size=4
                    )
                    
                    if eval_results.get("success"):
                        logger.info("Evaluation completed successfully!")
                        logger.info(f"  Test Loss: {eval_results.get('loss', 'N/A'):.4f}")
                        logger.info(f"  Test Perplexity: {eval_results.get('perplexity', 'N/A'):.2f}")
                        
                        # Save evaluation results
                        save_evaluation_results(
                            eval_results,
                            "reports/evaluation_results.json"
                        )
                        
                        # Generate sample predictions
                        logger.info("\nGenerating sample predictions...")
                        sample_predictions = generate_sample_predictions(
                            model_path=model_path,
                            max_new_tokens=50
                        )
                        
                        if sample_predictions:
                            logger.info("Sample predictions generated successfully!")
                            for pred in sample_predictions:
                                logger.info(f"  Prompt: {pred['prompt']}")
                                logger.info(f"  Generated: {pred['generated']}")
                    else:
                        logger.warning(f"Evaluation failed: {eval_results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}", exc_info=True)
                    logger.warning("Continuing despite evaluation errors...")
                
            else:
                logger.error(f"Fine-tuning failed: {fine_tuning_results.get('error', 'Unknown error')}")
        else:
            logger.warning("Skipping fine-tuning: Tokenization did not succeed")
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}", exc_info=True)
        logger.warning("Continuing despite fine-tuning errors...")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline execution complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
