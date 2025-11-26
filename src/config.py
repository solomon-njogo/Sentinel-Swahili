"""
Configuration Management for Transformer Fine-tuning
Handles hyperparameters, model configuration, and training settings.
"""

import json
import os
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Swahili-specific model options
SWAHILI_MODELS = {
    'davlan': 'Davlan/xlm-roberta-base-finetuned-swahili',
    'xlm-roberta': 'xlm-roberta-base',
    'xlm-roberta-large': 'xlm-roberta-large',
    'default': 'xlm-roberta-base'
}


class TrainingConfig:
    """
    Configuration class for transformer fine-tuning.
    Manages all hyperparameters and settings.
    """
    
    def __init__(
        self,
        # Model configuration
        model_name: str = "xlm-roberta-base",
        tokenizer_name: Optional[str] = None,
        max_length: int = 512,
        use_swahili_model: bool = False,
        swahili_model_key: str = 'davlan',  # Key from SWAHILI_MODELS dict
        
        # LoRA configuration (for reference/documentation)
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.1,
        
        # Training configuration
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        
        # Data configuration
        data_dir: str = "data",
        output_dir: str = "outputs",
        checkpoint_dir: str = "checkpoints",
        use_raw_text: bool = True,
        
        # Device configuration
        device: Optional[str] = None,  # 'cuda', 'cpu', or None for auto-detect
        use_cuda: bool = True,
        
        # Evaluation configuration
        eval_strategy: str = "epoch",  # 'epoch', 'steps', or 'no'
        eval_steps: Optional[int] = None,
        save_strategy: str = "epoch",  # 'epoch', 'steps', or 'no'
        save_steps: Optional[int] = None,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "accuracy",
        
        # Logging configuration
        logging_dir: str = "logs",
        logging_steps: int = 100,
        report_to: Optional[List[str]] = None,  # ['wandb', 'tensorboard', etc.]
        
        # Other configuration
        seed: int = 42,
        fp16: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False
    ):
        """
        Initialize training configuration.
        
        Args:
            model_name: Name of the pre-trained model to use
            tokenizer_name: Name of tokenizer (if different from model_name)
            max_length: Maximum sequence length for tokenization
            lora_rank: LoRA rank (number of low-rank dimensions)
            lora_alpha: LoRA alpha (scaling factor)
            lora_target_modules: List of module names to apply LoRA to (e.g., ['query', 'value'])
            lora_dropout: LoRA dropout rate
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps (0 to use warmup_ratio)
            warmup_ratio: Ratio of warmup steps to total steps
            max_grad_norm: Maximum gradient norm for clipping
            data_dir: Directory containing data files
            output_dir: Directory for outputs
            checkpoint_dir: Directory for model checkpoints
            use_raw_text: Whether to use raw text (True) or cleaned text (False)
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_cuda: Whether to use CUDA if available
            eval_strategy: When to evaluate ('epoch', 'steps', 'no')
            eval_steps: Evaluate every N steps (if eval_strategy='steps')
            save_strategy: When to save checkpoints ('epoch', 'steps', 'no')
            save_steps: Save every N steps (if save_strategy='steps')
            load_best_model_at_end: Whether to load best model at end of training
            metric_for_best_model: Metric to use for best model selection
            logging_dir: Directory for logs
            logging_steps: Log every N steps
            report_to: List of logging integrations to use
            seed: Random seed
            fp16: Whether to use mixed precision training
            dataloader_num_workers: Number of DataLoader worker processes
            dataloader_pin_memory: Whether to pin memory in DataLoader
        """
        # Model configuration
        # Use Swahili-specific model if requested
        if use_swahili_model and swahili_model_key in SWAHILI_MODELS:
            self.model_name = SWAHILI_MODELS[swahili_model_key]
            logger.info(f"Using Swahili-specific model: {self.model_name}")
        else:
            self.model_name = model_name
        
        self.tokenizer_name = tokenizer_name or self.model_name
        self.max_length = max_length
        self.use_swahili_model = use_swahili_model
        self.swahili_model_key = swahili_model_key
        
        # LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ['query', 'value', 'key', 'dense']
        self.lora_dropout = lora_dropout
        
        # Training configuration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        
        # Data configuration
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.use_raw_text = use_raw_text
        
        # Device configuration
        self.device = device
        self.use_cuda = use_cuda
        
        # Evaluation configuration
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        
        # Logging configuration
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.report_to = report_to or []
        
        # Other configuration
        self.seed = seed
        self.fp16 = fp16
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_name': self.model_name,
            'tokenizer_name': self.tokenizer_name,
            'max_length': self.max_length,
            'use_swahili_model': self.use_swahili_model,
            'swahili_model_key': self.swahili_model_key,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_target_modules': self.lora_target_modules,
            'lora_dropout': self.lora_dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'max_grad_norm': self.max_grad_norm,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'use_raw_text': self.use_raw_text,
            'device': self.device,
            'use_cuda': self.use_cuda,
            'eval_strategy': self.eval_strategy,
            'eval_steps': self.eval_steps,
            'save_strategy': self.save_strategy,
            'save_steps': self.save_steps,
            'load_best_model_at_end': self.load_best_model_at_end,
            'metric_for_best_model': self.metric_for_best_model,
            'logging_dir': self.logging_dir,
            'logging_steps': self.logging_steps,
            'report_to': self.report_to,
            'seed': self.seed,
            'fp16': self.fp16,
            'dataloader_num_workers': self.dataloader_num_workers,
            'dataloader_pin_memory': self.dataloader_pin_memory,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save the configuration file
        """
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to the configuration file
        
        Returns:
            TrainingConfig instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"TrainingConfig(model_name={self.model_name}, batch_size={self.batch_size}, lr={self.learning_rate})"


if __name__ == "__main__":
    # Test configuration
    print("=" * 80)
    print("Testing TrainingConfig")
    print("=" * 80)
    
    # Create default config
    config = TrainingConfig()
    print("\nDefault Configuration:")
    print(config)
    print(f"\nModel: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"LoRA alpha: {config.lora_alpha}")
    
    # Create custom config
    print("\n" + "=" * 80)
    print("Custom Configuration")
    print("=" * 80)
    custom_config = TrainingConfig(
        model_name="roberta-base",
        batch_size=32,
        learning_rate=3e-5,
        num_epochs=5,
        lora_rank=16,
        lora_alpha=32
    )
    print(custom_config)
    
    # Test save/load
    print("\n" + "=" * 80)
    print("Testing Save/Load")
    print("=" * 80)
    config_path = "test_config.json"
    custom_config.save(config_path)
    
    loaded_config = TrainingConfig.load(config_path)
    print(f"Loaded config: {loaded_config}")
    print(f"Model name matches: {loaded_config.model_name == custom_config.model_name}")
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"Cleaned up {config_path}")
    
    print("\n" + "=" * 80)
    print("TrainingConfig Test Complete!")
    print("=" * 80)

