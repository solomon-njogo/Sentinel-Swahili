"""
Standalone Training Script for Swahili Text Classification
Can be run independently after data preparation.
"""

import argparse
import os
import sys
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import modules
from transformers import DataCollatorForLanguageModeling

from src.config import TrainingConfig
from src.device_utils import get_device, print_device_info, set_seed
from src.tokenizer_utils import load_tokenizer
from src.transformer_preprocessing import prepare_transformer_data
from src.transformer_dataset import TransformerDataset
from src.model_trainer import (
    load_model_with_lora,
    setup_trainer,
    train_model,
    evaluate_model,
    compute_metrics
)
from src.causal_dataset import CausalLMChunkedDataset


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    """Main training execution function."""
    parser = argparse.ArgumentParser(
        description="Train Transformer Model for Swahili Text Classification"
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Path to configuration JSON file (optional)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing data files (default: data)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name to use (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--use-raw-text',
        action='store_true',
        default=True,
        help='Use raw text for transformers (default: True)'
    )
    parser.add_argument(
        '--use-swahili-model',
        action='store_true',
        help='Use Swahili-specific model'
    )
    parser.add_argument(
        '--swahili-model-key',
        type=str,
        default='davlan',
        help='Swahili model key (default: davlan)'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation on test set'
    )
    parser.add_argument(
        '--training-task',
        choices=['classification', 'mlm', 'causal'],
        default='classification',
        help='Choose training objective (default: classification)'
    )
    parser.add_argument(
        '--mlm-probability',
        type=float,
        default=0.15,
        help='Masking probability for MLM objective'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=None,
        help='Token block size for causal LM training'
    )
    parser.add_argument(
        '--block-stride',
        type=int,
        default=None,
        help='Stride (token overlap) between causal LM blocks'
    )
    parser.add_argument(
        '--no-causal-padding',
        action='store_true',
        help='Disable padding the final causal LM block'
    )
    parser.add_argument(
        '--no-shuffle-chunks',
        action='store_true',
        help='Keep causal LM blocks ordered (no shuffling)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("  SWAHILI TEXT CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)
    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load or create configuration
        print_section("Configuration Setup")
        if args.config_path and os.path.exists(args.config_path):
            print(f"\n  Loading configuration from: {args.config_path}")
            config = TrainingConfig.load(args.config_path)
        else:
            print(f"\n  Creating configuration...")
            config_kwargs = {}
            if args.model_name:
                config_kwargs['model_name'] = args.model_name
            if args.batch_size:
                config_kwargs['batch_size'] = args.batch_size
            if args.learning_rate:
                config_kwargs['learning_rate'] = args.learning_rate
            if args.num_epochs:
                config_kwargs['num_epochs'] = args.num_epochs
            config_kwargs['use_swahili_model'] = args.use_swahili_model
            config_kwargs['swahili_model_key'] = args.swahili_model_key
            config_kwargs['data_dir'] = args.data_dir
            config_kwargs['training_task'] = args.training_task
            config_kwargs['mlm_probability'] = args.mlm_probability
            if args.block_size:
                config_kwargs['block_size'] = args.block_size
            if args.block_stride is not None:
                config_kwargs['block_stride'] = args.block_stride
            if args.no_causal_padding:
                config_kwargs['causal_padding'] = False
            if args.no_shuffle_chunks:
                config_kwargs['shuffle_causal_chunks'] = False
            
            config = TrainingConfig(**config_kwargs)
        
        # Override config with command-line args if provided
        if args.model_name:
            config.model_name = args.model_name
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        if args.num_epochs:
            config.num_epochs = args.num_epochs
        if args.training_task:
            config.training_task = args.training_task
        if args.mlm_probability is not None:
            config.mlm_probability = args.mlm_probability
        if args.block_size:
            config.block_size = args.block_size
        if args.block_stride is not None:
            config.block_stride = args.block_stride
        if args.no_causal_padding:
            config.causal_padding = False
        if args.no_shuffle_chunks:
            config.shuffle_causal_chunks = False
        
        print(f"\n  Configuration:")
        print(f"    Model: {config.model_name}")
        print(f"    Batch size: {config.batch_size}")
        print(f"    Learning rate: {config.learning_rate}")
        print(f"    Epochs: {config.num_epochs}")
        print(f"    LoRA rank: {config.lora_rank}")
        print(f"    LoRA alpha: {config.lora_alpha}")
        print(f"    Training task: {config.training_task}")
        
        # Set up device
        print_section("Device Setup")
        set_seed(config.seed)
        device = get_device(use_cuda=config.use_cuda, device=config.device)
        print_device_info()
        
        # Load tokenizer
        print_section("Tokenizer Setup")
        print(f"\n  Loading tokenizer: {config.tokenizer_name}")
        tokenizer = load_tokenizer(
            model_name=config.tokenizer_name,
            use_fast=True
        )
        
        # Prepare data
        print_section("Data Preparation")
        print(f"\n  Preparing data from: {args.data_dir}")
        transformer_data = prepare_transformer_data(
            data_dir=args.data_dir,
            use_raw_text=args.use_raw_text
        )
        
        print(f"\n  Data Summary:")
        print(f"    Train samples: {len(transformer_data['texts_train']):,}")
        print(f"    Valid samples: {len(transformer_data['texts_valid']):,}")
        print(f"    Test samples:  {len(transformer_data['texts_test']):,}")
        
        if config.training_task == 'classification':
            if transformer_data['label_mapping']:
                num_classes = len(transformer_data['label_mapping'])
                print(f"    Number of classes: {num_classes}")
            else:
                print("    ERROR: No labels found for classification task!")
                sys.exit(1)
        elif config.training_task == 'causal':
            num_classes = None
            print("    Running causal language modeling (auto-regressive) training")
        else:
            num_classes = None
            print("    Running unsupervised MLM training (no labels required)")
        
        # Create datasets
        print_section("Dataset Creation")
        if config.training_task == 'causal':
            train_dataset = CausalLMChunkedDataset(
                texts=transformer_data['texts_train'],
                tokenizer=tokenizer,
                block_size=config.block_size,
                stride=config.block_stride,
                pad_to_block=config.causal_padding,
                shuffle_chunks=config.shuffle_causal_chunks
            )
        else:
            train_labels = transformer_data['labels_train'] if config.training_task == 'classification' else None
            train_dataset = TransformerDataset(
                texts=transformer_data['texts_train'],
                labels=train_labels,
                tokenizer=tokenizer,
                max_length=config.max_length
            )
        
        eval_dataset = None
        if transformer_data['texts_valid']:
            if config.training_task == 'causal':
                eval_dataset = CausalLMChunkedDataset(
                    texts=transformer_data['texts_valid'],
                    tokenizer=tokenizer,
                    block_size=config.block_size,
                    stride=config.block_stride,
                    pad_to_block=True,
                    shuffle_chunks=False
                )
            else:
                eval_labels = transformer_data['labels_valid'] if config.training_task == 'classification' else None
                eval_dataset = TransformerDataset(
                    texts=transformer_data['texts_valid'],
                    labels=eval_labels,
                    tokenizer=tokenizer,
                    max_length=config.max_length
                )
        
        test_dataset = None
        if transformer_data['texts_test'] and not args.skip_eval:
            if config.training_task == 'causal':
                test_dataset = CausalLMChunkedDataset(
                    texts=transformer_data['texts_test'],
                    tokenizer=tokenizer,
                    block_size=config.block_size,
                    stride=config.block_stride,
                    pad_to_block=True,
                    shuffle_chunks=False
                )
            else:
                test_labels = transformer_data['labels_test'] if config.training_task == 'classification' else None
                test_dataset = TransformerDataset(
                    texts=transformer_data['texts_test'],
                    labels=test_labels,
                    tokenizer=tokenizer,
                    max_length=config.max_length
                )
        
        print(f"\n  Datasets created:")
        print(f"    Train: {len(train_dataset):,} samples")
        if eval_dataset:
            print(f"    Valid: {len(eval_dataset):,} samples")
        if test_dataset:
            print(f"    Test:  {len(test_dataset):,} samples")
        
        # Load model with LoRA
        print_section("Model Loading")
        model = load_model_with_lora(
            config=config,
            num_labels=num_classes,
            device=device
        )
        
        # Set up trainer
        print_section("Trainer Setup")
        data_collator = None
        compute_metrics_fn = compute_metrics if config.training_task == 'classification' else None
        if config.training_task == 'mlm':
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=config.mlm_probability
            )
            # For MLM we care about minimizing loss
            config.metric_for_best_model = "eval_loss"
            config.metric_greater_is_better = False
        elif config.training_task == 'causal':
            data_collator = None  # Dataset already returns fixed-length tensors
            config.metric_for_best_model = "eval_loss"
            config.metric_greater_is_better = False
        
        trainer = setup_trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics_fn=compute_metrics_fn,
            data_collator=data_collator
        )
        
        # Train model
        print_section("Training")
        training_results = train_model(
            model=model,
            trainer=trainer,
            config=config
        )
        
        # Evaluate on test set
        if test_dataset and not args.skip_eval:
            print_section("Test Evaluation")
            test_metrics = evaluate_model(
                trainer=trainer,
                dataset=test_dataset,
                dataset_name="test"
            )
            training_results['test_metrics'] = test_metrics
        
        # Final summary
        print_section("Training Complete")
        print(f"\n  Training completed successfully!")
        print(f"\n  Final model saved to: {training_results['final_model_path']}")
        if training_results.get('best_model_path'):
            print(f"  Best model saved to: {training_results['best_model_path']}")
        
        if training_results.get('train_metrics'):
            print(f"\n  Training Metrics:")
            for key, value in training_results['train_metrics'].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
        
        if training_results.get('eval_metrics'):
            print(f"\n  Validation Metrics:")
            for key, value in training_results['eval_metrics'].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
        
        if training_results.get('test_metrics'):
            print(f"\n  Test Metrics:")
            for key, value in training_results['test_metrics'].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
        
        print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ERROR: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

