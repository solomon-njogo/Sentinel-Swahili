"""
Model Training Module for Transformer Fine-tuning with LoRA
Handles model loading, LoRA configuration, and training setup using Hugging Face Trainer.
"""

import logging
import os
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

from .config import TrainingConfig
from .evaluation_utils import calculate_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_with_lora(
    config: TrainingConfig,
    num_labels: Optional[int],
    device: torch.device
) -> torch.nn.Module:
    """
    Load base model and apply LoRA adapters.
    
    Args:
        config: Training configuration
        num_labels: Number of classification labels
        device: Device to load model on
    
    Returns:
        Model with LoRA adapters applied
    """
    logger.info(f"Loading base model: {config.model_name}")
    
    if config.training_task == "classification":
        if num_labels is None:
            raise ValueError("num_labels must be provided for classification task")
        logger.info(f"Number of classes: {num_labels}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
            torch_dtype=torch.float16 if config.fp16 and torch.cuda.is_available() else torch.float32,
        )
        lora_task_type = TaskType.SEQ_CLS
    elif config.training_task == "mlm":
        logger.info("Initializing model for masked language modeling")
        model = AutoModelForMaskedLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 and torch.cuda.is_available() else torch.float32,
        )
        lora_task_type = getattr(TaskType, "MASKED_LM", TaskType.TOKEN_CLS)
    else:
        logger.info("Initializing model for causal language modeling")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 and torch.cuda.is_available() else torch.float32,
        )
        lora_task_type = TaskType.CAUSAL_LM
    
    # Configure LoRA
    logger.info(f"Configuring LoRA with rank={config.lora_rank}, alpha={config.lora_alpha}")
    lora_config = LoraConfig(
        task_type=lora_task_type,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Move model to device
    model = model.to(device)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    logger.info("Model with LoRA adapters loaded successfully")
    return model


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
    
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Get predicted class labels
    predictions = np.argmax(predictions, axis=-1)
    
    # Calculate metrics
    metrics = calculate_all_metrics(labels, predictions, average='weighted')
    
    return metrics


def setup_trainer(
    model: torch.nn.Module,
    config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Optional[Any],
    tokenizer: Any,
    compute_metrics_fn: Optional[callable] = None,
    data_collator: Optional[Any] = None
) -> Trainer:
    """
    Set up Hugging Face Trainer with training arguments.
    
    Args:
        model: Model to train
        config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Validation dataset (optional)
        tokenizer: Tokenizer instance
        compute_metrics_fn: Function to compute metrics (optional)
    
    Returns:
        Configured Trainer instance
    """
    # Calculate total steps
    num_train_samples = len(train_dataset)
    steps_per_epoch = num_train_samples // config.batch_size
    if num_train_samples % config.batch_size != 0:
        steps_per_epoch += 1
    
    total_steps = steps_per_epoch * config.num_epochs
    
    # Calculate warmup steps
    if config.warmup_steps > 0:
        warmup_steps = config.warmup_steps
    else:
        warmup_steps = int(total_steps * config.warmup_ratio)
    
    logger.info(f"Training configuration:")
    logger.info(f"  Total samples: {num_train_samples:,}")
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Warmup steps: {warmup_steps:,}")
    
    # Set up evaluation strategy
    eval_strategy = config.eval_strategy
    eval_steps = config.eval_steps
    if eval_strategy == 'steps' and eval_steps is None:
        eval_steps = steps_per_epoch  # Evaluate once per epoch by default
    
    # Set up save strategy
    save_strategy = config.save_strategy
    save_steps = config.save_steps
    if save_strategy == 'steps' and save_steps is None:
        save_steps = steps_per_epoch  # Save once per epoch by default
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.metric_greater_is_better,
        save_total_limit=3,  # Keep only last 3 checkpoints
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        report_to=config.report_to if config.report_to else [],
        seed=config.seed,
        remove_unused_columns=False,  # Keep all columns from dataset
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        data_collator=data_collator,
    )
    
    logger.info("Trainer configured successfully")
    return trainer


def train_model(
    model: torch.nn.Module,
    trainer: Trainer,
    config: TrainingConfig
) -> Dict[str, Any]:
    """
    Train the model using the Trainer.
    
    Args:
        model: Model to train
        trainer: Configured Trainer instance
        config: Training configuration
    
    Returns:
        Dictionary with training results and metrics
    """
    logger.info("Starting training...")
    
    # Train the model
    train_result = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    # Get training metrics
    train_metrics = train_result.metrics
    
    # Evaluate on validation set if available
    eval_metrics = None
    if trainer.eval_dataset is not None:
        logger.info("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        logger.info(f"Validation metrics: {eval_metrics}")
    
    # Get best model path if load_best_model_at_end is True
    best_model_path = None
    if config.load_best_model_at_end and trainer.state.best_metric is not None:
        best_model_path = os.path.join(config.checkpoint_dir, f"checkpoint-{trainer.state.best_model_checkpoint}")
        logger.info(f"Best model checkpoint: {best_model_path}")
        logger.info(f"Best metric ({config.metric_for_best_model}): {trainer.state.best_metric:.4f}")
    
    results = {
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'best_model_path': best_model_path,
        'final_model_path': final_model_path,
        'training_complete': True
    }
    
    logger.info("Training completed successfully")
    return results


def evaluate_model(
    trainer: Trainer,
    dataset: Any,
    dataset_name: str = "test"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        trainer: Trainer instance with trained model
        dataset: Dataset to evaluate on
        dataset_name: Name of dataset for logging
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating on {dataset_name} dataset...")
    
    eval_results = trainer.evaluate(eval_dataset=dataset)
    
    logger.info(f"{dataset_name} metrics:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return eval_results


if __name__ == "__main__":
    # Test model trainer components
    print("=" * 80)
    print("Testing Model Trainer")
    print("=" * 80)
    
    # Create test config
    from .config import TrainingConfig
    test_config = TrainingConfig(
        model_name="xlm-roberta-base",
        batch_size=4,
        num_epochs=1,
        lora_rank=4,
        lora_alpha=8
    )
    
    print(f"\nTest Configuration:")
    print(f"  Model: {test_config.model_name}")
    print(f"  LoRA rank: {test_config.lora_rank}")
    print(f"  LoRA alpha: {test_config.lora_alpha}")
    print(f"  Batch size: {test_config.batch_size}")
    
    print("\n" + "=" * 80)
    print("Model Trainer Test Complete!")
    print("=" * 80)

