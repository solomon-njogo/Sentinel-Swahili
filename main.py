"""
Main Pipeline for Swahili Text Processing and Transformer Fine-tuning
Orchestrates the complete pipeline from data loading to model training.

Usage:
    # Data preparation only:
    python main.py
    
    # Full pipeline (data prep + training):
    python main.py --train
    
    # With custom configuration:
    python main.py --train --model-name xlm-roberta-base --batch-size 32 --num-epochs 5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from transformers import DataCollatorForLanguageModeling

# Suppress verbose logging during execution
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Import pipeline modules
from src.data_pipeline import DataPipeline
from src.data_inspector import DataInspector
from src.transformer_preprocessing import prepare_transformer_data
from src.config import TrainingConfig
from src.device_utils import get_device, print_device_info, set_seed
from src.tokenizer_utils import load_tokenizer, get_tokenizer_info
from src.transformer_dataset import TransformerDataset, create_data_loader
from src.model_trainer import load_model_with_lora, setup_trainer, train_model, evaluate_model, compute_metrics


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str, width: int = 80):
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def save_report(stats: dict, output_path: str):
    """Save statistics report to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n[✓] Report saved to: {output_path}")


def step1_data_loading_and_inspection(
    data_dir: str,
    files: list,
    output_dir: str,
    save_report_flag: bool,
    tokenizer: Optional[Any] = None
) -> Dict[str, Any]:
    """Step 1: Load and inspect raw data."""
    print_section("STEP 1: Data Loading and Inspection")
    
    pipeline = DataPipeline(data_dir=data_dir)
    inspector = DataInspector()
    all_results = {}
    
    for filename in files:
        print_subsection(f"Processing: {filename}")
        
        try:
            # Get basic file info
            file_info = pipeline.get_dataset_info(filename)
            print(f"\n  File Information:")
            for key, value in file_info.items():
                if key != "error":
                    if isinstance(value, float):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value:,}" if isinstance(value, int) else f"    {key}: {value}")
            
            if "error" in file_info:
                print(f"    ERROR: {file_info['error']}")
                continue
            
            # Parse dataset
            print(f"\n  Parsing dataset...")
            features, labels, label_format = pipeline.parse_dataset(filename)
            
            if not features:
                print(f"    WARNING: No data found in {filename}")
                continue
            
            # Inspect dataset (with optional tokenizer for tokenized length analysis)
            print(f"  Inspecting dataset...")
            stats = inspector.inspect_dataset(
                features, 
                labels if any(l is not None for l in labels) else None,
                tokenizer=tokenizer
            )
            
            # Add metadata
            stats["metadata"] = {
                "filename": filename,
                "label_format": label_format,
                "processing_date": datetime.now().isoformat(),
                "file_info": file_info
            }
            
            # Print summary
            dataset_name = filename.replace('.txt', '').upper()
            print(f"\n  Dataset Summary ({dataset_name}):")
            text_stats = stats.get('text_statistics', {})
            print(f"    Total samples: {text_stats.get('total_samples', len(features)):,}")
            if text_stats:
                print(f"    Avg characters: {text_stats.get('char_length', {}).get('mean', 0):.1f}")
                print(f"    Avg words: {text_stats.get('word_length', {}).get('mean', 0):.1f}")
            
            # Print tokenized length statistics if available
            if 'tokenized_length_statistics' in stats and stats['tokenized_length_statistics']:
                token_stats = stats['tokenized_length_statistics']
                print(f"\n  Tokenized Length Statistics (after transformer tokenization):")
                print(f"    Mean: {token_stats.get('mean', 0):.1f} tokens")
                print(f"    Median: {token_stats.get('median', 0):.1f} tokens")
                print(f"    P95: {token_stats.get('p95', 0):.1f} tokens")
                print(f"    Max: {token_stats.get('max', 0):.1f} tokens")
                recommended = token_stats.get('recommended_max_length', 512)
                print(f"    Recommended max_length: {recommended} tokens")
            
            # Store results
            all_results[filename] = stats
            
            # Save individual report if requested
            if save_report_flag:
                report_path = os.path.join(
                    output_dir,
                    f"{filename.replace('.txt', '')}_report.json"
                )
                save_report(stats, report_path)
        
        except FileNotFoundError as e:
            print(f"    ERROR: {e}")
        except Exception as e:
            print(f"    ERROR processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined report if requested
    if save_report_flag and all_results:
        combined_report_path = os.path.join(output_dir, "combined_report.json")
        combined_report = {
            "processing_date": datetime.now().isoformat(),
            "datasets": all_results,
            "summary": {
                "total_datasets": len(all_results),
                "dataset_names": list(all_results.keys())
            }
        }
        save_report(combined_report, combined_report_path)
    
    print(f"\n[✓] Step 1 Complete: Data loaded and inspected")
    return all_results


def step2_transformer_data_preparation(
    data_dir: str,
    use_raw_text: bool = True
) -> Dict[str, Any]:
    """Step 2: Prepare data for transformer models."""
    print_section("STEP 2: Transformer Data Preparation")
    
    print(f"\n  Preparing data for transformer models...")
    print(f"  Using raw text (recommended for transformers): {use_raw_text}")
    
    transformer_data = prepare_transformer_data(
        data_dir=data_dir,
        use_raw_text=use_raw_text
    )
    
    print(f"\n  Data Preparation Summary:")
    print(f"    Train samples: {len(transformer_data['texts_train']):,}")
    print(f"    Test samples:  {len(transformer_data['texts_test']):,}")
    print(f"    Valid samples: {len(transformer_data['texts_valid']):,}")
    
    if transformer_data['label_mapping']:
        num_classes = len(transformer_data['label_mapping'])
        print(f"    Number of classes: {num_classes}")
        print(f"    Label mapping: {transformer_data['label_mapping']}")
    else:
        print(f"    Labels: None (unlabeled dataset)")
    
    print(f"\n[✓] Step 2 Complete: Data prepared for transformers")
    return transformer_data


def step3_configuration_setup(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    use_swahili_model: bool = False,
    swahili_model_key: str = 'davlan',
    training_task: Optional[str] = None,
    mlm_probability: Optional[float] = None
) -> TrainingConfig:
    """Step 3: Set up training configuration."""
    print_section("STEP 3: Configuration Setup")
    
    if config_path and os.path.exists(config_path):
        print(f"\n  Loading configuration from: {config_path}")
        config = TrainingConfig.load(config_path)
    else:
        print(f"\n  Creating default configuration...")
        config_kwargs = {}
        if model_name:
            config_kwargs['model_name'] = model_name
        if batch_size:
            config_kwargs['batch_size'] = batch_size
        if learning_rate:
            config_kwargs['learning_rate'] = learning_rate
        config_kwargs['use_swahili_model'] = use_swahili_model
        config_kwargs['swahili_model_key'] = swahili_model_key
        if training_task:
            config_kwargs['training_task'] = training_task
        if mlm_probability is not None:
            config_kwargs['mlm_probability'] = mlm_probability
        
        config = TrainingConfig(**config_kwargs)
    
    if training_task:
        config.training_task = training_task
    if mlm_probability is not None:
        config.mlm_probability = mlm_probability
    
    print(f"\n  Configuration Summary:")
    print(f"    Model: {config.model_name}")
    if config.use_swahili_model:
        print(f"    Swahili-specific model: Yes ({config.swahili_model_key})")
    print(f"    Tokenizer: {config.tokenizer_name}")
    print(f"    Max length: {config.max_length}")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Epochs: {config.num_epochs}")
    print(f"    LoRA rank: {config.lora_rank}")
    print(f"    LoRA alpha: {config.lora_alpha}")
    print(f"    Data directory: {config.data_dir}")
    print(f"    Output directory: {config.output_dir}")
    print(f"    Checkpoint directory: {config.checkpoint_dir}")
    print(f"    Training task: {config.training_task}")
    if config.training_task == 'mlm':
        print(f"    MLM probability: {config.mlm_probability}")
    
    # Save config if path provided
    if config_path:
        config.save(config_path)
        print(f"\n  ✓ Configuration saved to: {config_path}")
    
    print(f"\n[✓] Step 3 Complete: Configuration ready")
    return config


def step4_device_setup(config: TrainingConfig) -> Any:
    """Step 4: Set up device and environment."""
    print_section("STEP 4: Device and Environment Setup")
    
    # Set random seed
    print(f"\n  Setting random seed: {config.seed}")
    set_seed(config.seed)
    
    # Get device
    print(f"\n  Detecting device...")
    device = get_device(use_cuda=config.use_cuda, device=config.device)
    
    # Print device info
    print_device_info()
    
    print(f"\n[✓] Step 4 Complete: Device ready ({device})")
    return device


def step5_tokenizer_setup(config: TrainingConfig) -> Any:
    """Step 5: Load and configure tokenizer."""
    print_section("STEP 5: Tokenizer Setup")
    
    print(f"\n  Loading tokenizer: {config.tokenizer_name}")
    tokenizer = load_tokenizer(
        model_name=config.tokenizer_name,
        use_fast=True
    )
    
    # Get tokenizer info
    from src.tokenizer_utils import explain_special_tokens
    tokenizer_info = get_tokenizer_info(tokenizer)
    print(f"\n  Tokenizer Information:")
    print(f"    Model: {tokenizer_info['model_name']}")
    print(f"    Vocab size: {tokenizer_info['vocab_size']:,}")
    print(f"    Max length: {tokenizer_info['model_max_length']:,}")
    print(f"    Pad token: {tokenizer_info['pad_token']}")
    print(f"    Fast tokenizer: {tokenizer_info['is_fast']}")
    
    # Explain special tokens
    special_tokens = explain_special_tokens(tokenizer)
    if special_tokens:
        print(f"\n  Special Tokens:")
        for key, info in special_tokens.items():
            print(f"    {key}: {info['token']} - {info['name']} ({info['purpose']})")
    
    print(f"\n[✓] Step 5 Complete: Tokenizer ready")
    return tokenizer


def step6_dataset_creation(
    transformer_data: Dict[str, Any],
    tokenizer: Any,
    config: TrainingConfig
) -> Dict[str, Any]:
    """Step 6: Create PyTorch datasets and data loaders."""
    print_section("STEP 6: Dataset and DataLoader Creation")
    
    datasets = {}
    data_loaders = {}
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        texts_key = f'texts_{split}'
        labels_key = f'labels_{split}'
        
        if texts_key not in transformer_data or not transformer_data[texts_key]:
            print(f"\n  Skipping {split} (no data)")
            continue
        
        print(f"\n  Creating {split} dataset...")
        
        texts = transformer_data[texts_key]
        labels = transformer_data.get(labels_key) if config.training_task == 'classification' else None
        
        # Create dataset
        dataset = TransformerDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        
        datasets[split] = dataset
        
        # Create data loader
        shuffle = (split == 'train')
        data_loader = create_data_loader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory
        )
        
        data_loaders[split] = data_loader
        
        print(f"    Samples: {len(dataset):,}")
        print(f"    Batches: {len(data_loader):,}")
        if dataset.has_labels:
            print(f"    Classes: {dataset.get_num_classes()}")
        elif config.training_task == 'mlm':
            print(f"    Objective: Masked language modeling")
    
    print(f"\n[✓] Step 6 Complete: Datasets and DataLoaders ready")
    return {
        'datasets': datasets,
        'data_loaders': data_loaders
    }


def step7_final_summary(
    config: TrainingConfig,
    transformer_data: Dict[str, Any],
    datasets: Dict[str, Any],
    device: Any,
    tokenizer: Any
):
    """Step 7: Print final summary of everything ready for model loading."""
    print_section("STEP 7: Final Summary - Ready for Model Loading")
    
    print(f"\n  All components are ready for transformer fine-tuning!")
    
    print(f"\n  Data Summary:")
    print(f"    Train: {len(transformer_data['texts_train']):,} samples")
    print(f"    Valid: {len(transformer_data['texts_valid']):,} samples")
    print(f"    Test:  {len(transformer_data['texts_test']):,} samples")
    
    if transformer_data['label_mapping']:
        print(f"    Classes: {len(transformer_data['label_mapping'])}")
    
    print(f"\n  Model Configuration:")
    print(f"    Base model: {config.model_name}")
    print(f"    Max sequence length: {config.max_length}")
    print(f"    LoRA rank: {config.lora_rank}")
    print(f"    LoRA alpha: {config.lora_alpha}")
    
    print(f"\n  Training Configuration:")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Epochs: {config.num_epochs}")
    print(f"    Device: {device}")
    print(f"    Training task: {config.training_task}")
    if config.training_task == 'mlm':
        print(f"    MLM probability: {config.mlm_probability}")
    
    print(f"\n  Tokenizer:")
    tokenizer_info = get_tokenizer_info(tokenizer)
    print(f"    Model: {tokenizer_info['model_name']}")
    print(f"    Vocab size: {tokenizer_info['vocab_size']:,}")
    
    print(f"\n  Datasets:")
    for split, dataset in datasets['datasets'].items():
        print(f"    {split}: {len(dataset):,} samples, {len(datasets['data_loaders'][split]):,} batches")
    
    print(f"\n  Next Steps:")
    print(f"    1. Load pre-trained model: {config.model_name}")
    print(f"    2. Configure LoRA adapter with rank={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"    3. Set up optimizer with lr={config.learning_rate}")
    print(f"    4. Begin training loop using the prepared DataLoaders")
    
    print(f"\n" + "=" * 80)
    print(f"  [✓] ALL PREPARATION STEPS COMPLETE")
    print(f"  [✓] READY FOR MODEL LOADING AND TRAINING")
    print("=" * 80 + "\n")


def step8_model_loading(
    config: TrainingConfig,
    transformer_data: Dict[str, Any],
    device: Any
) -> Any:
    """Step 8: Load model with LoRA adapters."""
    print_section("STEP 8: Model Loading with LoRA")
    
    num_classes = None
    print(f"\n  Loading model: {config.model_name}")

    if config.training_task == 'classification':
        if not transformer_data.get('label_mapping'):
            print("\n  ERROR: No labels found in dataset. Cannot train classification model.")
            return None
        num_classes = len(transformer_data['label_mapping'])
        print(f"  Number of classes: {num_classes}")
    else:
        print("  Objective: Masked language modeling (unsupervised)")
        print(f"  MLM probability: {config.mlm_probability}")

    print(f"  LoRA configuration:")
    print(f"    Rank: {config.lora_rank}")
    print(f"    Alpha: {config.lora_alpha}")
    print(f"    Target modules: {config.lora_target_modules}")
    
    model = load_model_with_lora(
        config=config,
        num_labels=num_classes,
        device=device
    )
    
    print(f"\n[✓] Step 8 Complete: Model loaded with LoRA adapters")
    return model


def step9_training(
    model: Any,
    config: TrainingConfig,
    datasets: Dict[str, Any],
    tokenizer: Any,
    transformer_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Step 9: Train the model."""
    print_section("STEP 9: Model Training")
    
    # Get datasets
    train_dataset = datasets['datasets'].get('train')
    eval_dataset = datasets['datasets'].get('valid')
    test_dataset = datasets['datasets'].get('test')
    
    if not train_dataset:
        print("\n  ERROR: No training dataset available.")
        return {}
    
    print(f"\n  Training Configuration:")
    print(f"    Train samples: {len(train_dataset):,}")
    if eval_dataset:
        print(f"    Valid samples: {len(eval_dataset):,}")
    if test_dataset:
        print(f"    Test samples:  {len(test_dataset):,}")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Epochs: {config.num_epochs}")
    
    # Set up trainer
    print(f"\n  Setting up trainer...")
    data_collator = None
    compute_metrics_fn = compute_metrics if config.training_task == 'classification' else None
    if config.training_task == 'mlm':
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=config.mlm_probability
        )
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
    print(f"\n  Starting training...")
    training_results = train_model(
        model=model,
        trainer=trainer,
        config=config
    )
    
    # Evaluate on test set if available
    if test_dataset and (config.training_task == 'mlm' or test_dataset.has_labels):
        print(f"\n  Evaluating on test set...")
        test_metrics = evaluate_model(
            trainer=trainer,
            dataset=test_dataset,
            dataset_name="test"
        )
        training_results['test_metrics'] = test_metrics
    
    # Print summary
    print(f"\n  Training Results:")
    if training_results.get('train_metrics'):
        train_metrics = training_results['train_metrics']
        for key, value in train_metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
    if training_results.get('eval_metrics'):
        eval_metrics = training_results['eval_metrics']
        for key, value in eval_metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
    if training_results.get('test_metrics'):
        test_metrics = training_results['test_metrics']
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
    
    if training_results.get('best_model_path'):
        print(f"\n  Best model saved to: {training_results['best_model_path']}")
    if training_results.get('final_model_path'):
        print(f"  Final model saved to: {training_results['final_model_path']}")
    
    print(f"\n[✓] Step 9 Complete: Model training finished")
    return training_results


def main():
    """Main execution function - orchestrates the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Swahili Text Processing Pipeline - Complete Pipeline from Data to Model-Ready Datasets"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing data files (default: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Directory to save reports (default: reports)'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Path to load/save configuration JSON (optional)'
    )
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        default=['train.txt', 'test.txt', 'valid.txt'],
        help='List of data files to process (default: train.txt test.txt valid.txt)'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detailed reports to JSON files'
    )
    parser.add_argument(
        '--use-raw-text',
        action='store_true',
        default=True,
        help='Use raw text for transformers (default: True, recommended)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name to use (default: xlm-roberta-base)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model after data preparation (default: False)'
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
        help='Skip evaluation on test set during training'
    )
    parser.add_argument(
        '--training-task',
        choices=['classification', 'mlm'],
        default='classification',
        help='Choose training objective (default: classification)'
    )
    parser.add_argument(
        '--mlm-probability',
        type=float,
        default=0.15,
        help='Masking probability for MLM objective'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.save_report:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Print header
    print("\n" + "=" * 80)
    print("  SWAHILI TEXT PROCESSING PIPELINE")
    print("  Complete Pipeline: Data Loading -> Model-Ready Datasets")
    print("=" * 80)
    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 3: Configuration Setup (moved earlier to get model name for tokenizer)
        config = step3_configuration_setup(
            config_path=args.config_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_swahili_model=args.use_swahili_model,
            swahili_model_key=args.swahili_model_key,
            training_task=args.training_task,
            mlm_probability=args.mlm_probability
        )
        
        # Step 5: Tokenizer Setup (moved earlier for use in Step 1)
        tokenizer = step5_tokenizer_setup(config)
        
        # Step 1: Data Loading and Inspection (now with tokenizer for tokenized length analysis)
        inspection_results = step1_data_loading_and_inspection(
            data_dir=args.data_dir,
            files=args.files,
            output_dir=args.output_dir,
            save_report_flag=args.save_report,
            tokenizer=tokenizer
        )
        
        # Step 2: Transformer Data Preparation
        transformer_data = step2_transformer_data_preparation(
            data_dir=args.data_dir,
            use_raw_text=args.use_raw_text
        )
        
        # Step 4: Device Setup
        device = step4_device_setup(config)
        
        # Step 6: Dataset Creation
        dataset_dict = step6_dataset_creation(
            transformer_data=transformer_data,
            tokenizer=tokenizer,
            config=config
        )
        
        # Step 7: Final Summary
        step7_final_summary(
            config=config,
            transformer_data=transformer_data,
            datasets=dataset_dict,
            device=device,
            tokenizer=tokenizer
        )
        
        # Steps 8-9: Model Loading and Training (if --train flag is set)
        if args.train:
            model = step8_model_loading(
                config=config,
                transformer_data=transformer_data,
                device=device
            )
            
            if model is not None:
                training_results = step9_training(
                    model=model,
                    config=config,
                    datasets=dataset_dict,
                    tokenizer=tokenizer,
                    transformer_data=transformer_data
                )
                
                print("\n" + "=" * 80)
                print("  [✓] COMPLETE PIPELINE FINISHED")
                print("  [✓] DATA PREPROCESSING -> MODEL TRAINING")
                print("=" * 80 + "\n")
        else:
            print("\n  Note: Use --train flag to train the model after data preparation")
            print("  Example: python main.py --train\n")
        
        print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
