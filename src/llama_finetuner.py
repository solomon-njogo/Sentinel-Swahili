"""
Llama 2 Fine-tuning Module with QLoRA
Handles QLoRA (4-bit quantized LoRA) fine-tuning of Llama 2 7B model for Swahili text.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_from_disk
from huggingface_hub import login

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_llama_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    use_4bit: bool = True,
    device_map: str = "auto"
) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Load Llama 2 model with 4-bit quantization and tokenizer.
    
    Args:
        model_name: Name of the Llama 2 model (default: "meta-llama/Llama-2-7b-hf")
        use_4bit: Whether to use 4-bit quantization (default: True)
        device_map: Device mapping strategy (default: "auto")
        
    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading fails
    """
    logger.info(f"Loading Llama 2 model: {model_name}")
    logger.info(f"Using 4-bit quantization: {use_4bit}")
    
    # Authenticate with Hugging Face Hub
    try:
        login(new_session=False)
        logger.info("Authenticated with Hugging Face Hub")
    except KeyboardInterrupt:
        logger.error("Authentication interrupted by user")
        return None, None
    except Exception as e:
        logger.warning(f"Could not authenticate with Hugging Face Hub: {e}")
        logger.warning("Will attempt to use cached credentials or local files")
    
    # Check for local model paths
    local_llama_paths = [
        "./models/llama-2-7b-hf",
        "./models/llama2-7b",
        os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf"),
    ]
    
    model_path = None
    for local_path in local_llama_paths:
        expanded_path = os.path.expanduser(local_path)
        # Normalize path to handle mixed separators (especially on Windows)
        expanded_path = os.path.normpath(expanded_path)
        if os.path.exists(expanded_path):
            # Verify the directory contains necessary files (config.json or tokenizer.json)
            required_files = ["config.json", "tokenizer.json"]
            has_required = any(os.path.exists(os.path.join(expanded_path, f)) for f in required_files)
            if has_required:
                model_path = str(Path(expanded_path).resolve())
                logger.info(f"Found local model at: {model_path}")
                break
            else:
                logger.warning(f"Found directory at {expanded_path} but missing required model files. Skipping.")
    
    if model_path is None:
        model_path = model_name
        logger.info("Using Hugging Face Hub model (will download if not cached)")
    
    # Ensure model_path is a valid string
    if model_path is None or not isinstance(model_path, str) or not model_path.strip():
        logger.error(f"Invalid model_path: {model_path}")
        return None, None
    
    # Normalize path if it's a local path (not a Hugging Face model name)
    # Only normalize if it's actually a file path (contains separators or starts with .)
    if isinstance(model_path, str) and (os.path.sep in model_path or model_path.startswith('.')):
        try:
            model_path = str(Path(model_path).resolve())
        except (OSError, ValueError) as e:
            logger.warning(f"Could not resolve path {model_path}: {e}. Using as-is.")
    
    try:
        # Validate model_path before attempting to load
        if not model_path or not isinstance(model_path, str):
            raise ValueError(f"Invalid model_path: {model_path}")
        
        # Load tokenizer with retry logic
        logger.info(f"Loading tokenizer from: {model_path}")
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=True,
                    trust_remote_code=False,
                    resume_download=True
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries - 1 and ("connection" in error_str or "incomplete" in error_str or "broken" in error_str):
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Connection error loading tokenizer (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Check if bitsandbytes is available for 4-bit quantization
        bitsandbytes_available = False
        if use_4bit:
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                bitsandbytes_available = True
            except ImportError:
                logger.warning("=" * 80)
                logger.warning("bitsandbytes is not available. Cannot use 4-bit quantization.")
                logger.warning("Falling back to full precision (FP16) model loading.")
                logger.warning("")
                logger.warning("To enable 4-bit quantization, install bitsandbytes:")
                logger.warning("  - Linux: pip install bitsandbytes")
                logger.warning("  - Windows: bitsandbytes has limited support. Consider using WSL or Linux.")
                logger.warning("=" * 80)
                use_4bit = False
        
        # Load model with 4-bit quantization and retry logic
        max_retries = 3
        retry_delay = 10  # seconds (longer for model downloads)
        
        if use_4bit and bitsandbytes_available:
            logger.info("Loading model with 4-bit quantization (QLoRA)...")
            # BitsAndBytesConfig is already imported in the try block above
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            logger.info(f"Loading model from: {model_path}")
            for attempt in range(max_retries):
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=bnb_config,
                        device_map=device_map,
                        trust_remote_code=False,
                        torch_dtype=torch.float16,
                        resume_download=True,
                        low_cpu_mem_usage=True
                    )
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if attempt < max_retries - 1 and ("connection" in error_str or "incomplete" in error_str or "broken" in error_str or "read" in error_str):
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Connection error loading model (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
        else:
            logger.info("Loading model in full precision (FP16)...")
            logger.info(f"Loading model from: {model_path}")
            for attempt in range(max_retries):
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=device_map,
                        trust_remote_code=False,
                        torch_dtype=torch.float16,
                        resume_download=True,
                        low_cpu_mem_usage=True
                    )
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if attempt < max_retries - 1 and ("connection" in error_str or "incomplete" in error_str or "broken" in error_str or "read" in error_str):
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Connection error loading model (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
        
        logger.info("Model and tokenizer loaded successfully!")
        return model, tokenizer
        
    except ImportError as e:
        error_msg = str(e)
        logger.error(f"Missing dependency: {error_msg}")
        if "protobuf" in error_msg.lower():
            logger.error("=" * 80)
            logger.error("PROTOBUF ERROR: The protobuf library is required but not installed.")
            logger.error("Please install it with: pip install protobuf")
            logger.error("=" * 80)
        elif "bitsandbytes" in error_msg.lower() or "bitsandbytes" in str(e).lower():
            logger.error("=" * 80)
            logger.error("BITSANDBYTES ERROR: bitsandbytes is required for 4-bit quantization.")
            logger.error("Installation options:")
            logger.error("  - Linux: pip install bitsandbytes")
            logger.error("  - Windows: bitsandbytes has limited support. Consider:")
            logger.error("    1. Using WSL (Windows Subsystem for Linux)")
            logger.error("    2. Running on a Linux machine")
            logger.error("    3. Using full precision (set use_4bit=False)")
            logger.error("=" * 80)
        logger.error("Full error:", exc_info=True)
        return None, None
    except Exception as e:
        error_str = str(e).lower()
        
        # If loading from local path failed with path-related error, try Hugging Face Hub
        if ("endswith" in error_str or "nonetype" in error_str) and model_path != model_name:
            logger.warning(f"Failed to load from local path: {model_path}")
            logger.warning("Attempting to load from Hugging Face Hub instead...")
            try:
                # Retry with Hugging Face model name
                max_retries = 3
                retry_delay = 5
                for attempt in range(max_retries):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            use_fast=True,
                            trust_remote_code=False,
                            resume_download=True
                        )
                        break
                    except Exception as e:
                        error_str = str(e).lower()
                        if attempt < max_retries - 1 and ("connection" in error_str or "incomplete" in error_str or "broken" in error_str):
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"Connection error loading tokenizer (attempt {attempt + 1}/{max_retries}): {e}")
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            raise
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info("Set pad_token to eos_token")
                
                # Check if bitsandbytes is available for fallback
                bitsandbytes_available = False
                fallback_use_4bit = use_4bit
                if use_4bit:
                    try:
                        import bitsandbytes as bnb
                        from transformers import BitsAndBytesConfig
                        bitsandbytes_available = True
                    except ImportError:
                        logger.warning("bitsandbytes not available in fallback. Using full precision.")
                        fallback_use_4bit = False
                
                max_retries = 3
                retry_delay = 10
                
                if fallback_use_4bit and bitsandbytes_available:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    for attempt in range(max_retries):
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                quantization_config=bnb_config,
                                device_map=device_map,
                                trust_remote_code=False,
                                torch_dtype=torch.float16,
                                resume_download=True,
                                low_cpu_mem_usage=True
                            )
                            break
                        except Exception as e:
                            error_str = str(e).lower()
                            if attempt < max_retries - 1 and ("connection" in error_str or "incomplete" in error_str or "broken" in error_str or "read" in error_str):
                                wait_time = retry_delay * (2 ** attempt)
                                logger.warning(f"Connection error loading model (attempt {attempt + 1}/{max_retries}): {e}")
                                logger.info(f"Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                raise
                else:
                    for attempt in range(max_retries):
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                device_map=device_map,
                                trust_remote_code=False,
                                torch_dtype=torch.float16,
                                resume_download=True,
                                low_cpu_mem_usage=True
                            )
                            break
                        except Exception as e:
                            error_str = str(e).lower()
                            if attempt < max_retries - 1 and ("connection" in error_str or "incomplete" in error_str or "broken" in error_str or "read" in error_str):
                                wait_time = retry_delay * (2 ** attempt)
                                logger.warning(f"Connection error loading model (attempt {attempt + 1}/{max_retries}): {e}")
                                logger.info(f"Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                raise
                
                logger.info("Successfully loaded model from Hugging Face Hub!")
                return model, tokenizer
            except Exception as fallback_error:
                logger.error(f"Fallback to Hugging Face Hub also failed: {fallback_error}")
        
        logger.error(f"Failed to load model: {e}", exc_info=True)
        # Provide helpful error messages for common issues
        if "connection" in error_str or "incomplete" in error_str or "broken" in error_str:
            logger.error("=" * 80)
            logger.error("CONNECTION ERROR: Model download was interrupted.")
            logger.error("The model files are large (~13GB) and may take time to download.")
            logger.error("")
            logger.error("Solutions:")
            logger.error("  1. Check your internet connection")
            logger.error("  2. The download will automatically resume on the next run")
            logger.error("  3. For faster downloads, install hf_transfer:")
            logger.error("     pip install hf_transfer")
            logger.error("  4. Or use a VPN if you're experiencing connection issues")
            logger.error("=" * 80)
        elif "protobuf" in error_str:
            logger.error("=" * 80)
            logger.error("PROTOBUF ERROR: The protobuf library is required but not installed.")
            logger.error("Please install it with: pip install protobuf")
            logger.error("=" * 80)
        elif "endswith" in error_str or "nonetype" in error_str:
            logger.error("=" * 80)
            logger.error("PATH ERROR: Model path resolution failed.")
            logger.error(f"Attempted to load from: {model_path}")
            logger.error("This usually means:")
            logger.error("  1. The local model directory is incomplete or corrupted")
            logger.error("  2. Required files (config.json, tokenizer.json) are missing")
            logger.error("  3. The model path is invalid")
            logger.error("=" * 80)
        elif "cuda" in error_str or "gpu" in error_str:
            logger.warning("GPU-related error. If you don't have a GPU, the model will use CPU (slower).")
        return None, None


def setup_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None
) -> LoraConfig:
    """
    Configure LoRA parameters.
    
    Args:
        r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling parameter (default: 32)
        lora_dropout: LoRA dropout rate (default: 0.05)
        target_modules: List of module names to apply LoRA to (default: attention layers)
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    logger.info(f"Configuring LoRA with r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    return lora_config


def prepare_model_for_training(model, use_4bit: bool = True):
    """
    Prepare model for training with gradient checkpointing and LoRA.
    
    Args:
        model: The model to prepare
        use_4bit: Whether model uses 4-bit quantization
        
    Returns:
        Prepared model
    """
    logger.info("Preparing model for training...")
    
    if use_4bit:
        logger.info("Preparing 4-bit model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    logger.info("Model prepared for training")
    return model


def load_tokenized_datasets(
    tokenized_dir: str = "data/tokenized"
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """
    Load tokenized datasets from disk.
    
    Args:
        tokenized_dir: Directory containing tokenized datasets
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    tokenized_path = Path(tokenized_dir)
    
    datasets = {}
    for split in ["train", "valid", "test"]:
        split_path = tokenized_path / split
        if split_path.exists():
            try:
                logger.info(f"Loading {split} dataset from {split_path}...")
                dataset = load_from_disk(str(split_path))
                datasets[split] = dataset
                logger.info(f"Loaded {split} dataset: {len(dataset):,} examples")
            except Exception as e:
                logger.error(f"Failed to load {split} dataset: {e}")
                datasets[split] = None
        else:
            logger.warning(f"Tokenized {split} dataset not found at {split_path}")
            datasets[split] = None
    
    return datasets.get("train"), datasets.get("valid"), datasets.get("test")


def format_dataset_for_training(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Format dataset for causal language modeling.
    
    Args:
        dataset: Dataset with 'input_ids' feature
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Formatted dataset with 'input_ids' and 'labels'
    """
    def format_function(examples):
        # For causal LM, labels are the same as input_ids
        return {
            "input_ids": examples["input_ids"],
            "labels": examples["input_ids"]
        }
    
    logger.info(f"Formatting dataset for training (max_length={max_length})...")
    formatted = dataset.map(
        format_function,
        batched=False,
        remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "labels"]]
    )
    
    logger.info(f"Formatted dataset: {len(formatted):,} examples")
    return formatted


def fine_tune_llama(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    tokenized_dir: str = "data/tokenized",
    output_dir: str = "models/finetuned-llama2-7b",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    warmup_steps: int = 100
) -> Dict[str, Any]:
    """
    Fine-tune Llama 2 model with QLoRA.
    
    Args:
        model_name: Name of the base model
        tokenized_dir: Directory containing tokenized datasets
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_4bit: Whether to use 4-bit quantization
        save_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
        logging_steps: Steps between logging
        warmup_steps: Number of warmup steps
        
    Returns:
        Dictionary with training results and metrics
    """
    logger.info("=" * 80)
    logger.info("Starting Llama 2 Fine-tuning with QLoRA")
    logger.info("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer = load_llama_model_and_tokenizer(
        model_name=model_name,
        use_4bit=use_4bit
    )
    
    if model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer. Fine-tuning aborted.")
        return {"success": False, "error": "Model/tokenizer loading failed"}
    
    # Load datasets
    train_dataset, valid_dataset, test_dataset = load_tokenized_datasets(
        tokenized_dir=tokenized_dir
    )
    
    if train_dataset is None:
        logger.error("Training dataset not found. Fine-tuning aborted.")
        return {"success": False, "error": "Training dataset not found"}
    
    # Format datasets
    train_dataset = format_dataset_for_training(train_dataset, tokenizer, max_length)
    
    if valid_dataset is not None:
        valid_dataset = format_dataset_for_training(valid_dataset, tokenizer, max_length)
        eval_dataset = valid_dataset
    else:
        logger.warning("Validation dataset not found. Training without validation.")
        eval_dataset = None
    
    # Prepare model for training
    model = prepare_model_for_training(model, use_4bit=use_4bit)
    
    # Setup LoRA
    lora_config = setup_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    model = get_peft_model(model, lora_config)
    logger.info("LoRA adapters added to model")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Setup training arguments
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,  # Use mixed precision
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        warmup_steps=warmup_steps,
        report_to="none",  # Disable wandb/tensorboard
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        gradient_checkpointing=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Training examples: {len(train_dataset):,}")
    if eval_dataset:
        logger.info(f"Validation examples: {len(eval_dataset):,}")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    
    try:
        train_result = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save final model
        logger.info(f"Saving final model to {output_path}...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_path))
        
        # Evaluate on test set if available
        test_metrics = None
        if test_dataset is not None:
            logger.info("Evaluating on test set...")
            test_dataset = format_dataset_for_training(test_dataset, tokenizer, max_length)
            test_metrics = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(f"Test loss: {test_metrics.get('eval_loss', 'N/A'):.4f}")
        
        # Get training history
        training_history = trainer.state.log_history if hasattr(trainer.state, 'log_history') else []
        
        return {
            "success": True,
            "training_loss": train_result.training_loss,
            "test_metrics": test_metrics,
            "training_history": training_history,
            "output_dir": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

