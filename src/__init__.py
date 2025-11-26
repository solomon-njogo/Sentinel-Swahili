"""
Swahili Text Processing Pipeline
A comprehensive data engineering and preprocessing pipeline for Swahili text data.
"""

__version__ = "1.0.0"

# Original pipeline modules
from .data_pipeline import DataPipeline
from .data_inspector import DataInspector
from .text_cleaner import SwahiliTextCleaner
from .dataset_splitter import load_datasets, encode_labels

# Transformer fine-tuning utilities
from .tokenizer_utils import (
    load_tokenizer,
    tokenize_text,
    create_attention_mask,
    get_tokenizer_info,
    explain_special_tokens
)
from .transformer_dataset import (
    TransformerDataset,
    create_data_loader
)
from .transformer_preprocessing import (
    prepare_transformer_data,
    prepare_from_existing_datasets
)
from .config import TrainingConfig
from .device_utils import (
    get_device,
    move_to_device,
    get_device_info,
    print_device_info,
    clear_gpu_cache,
    set_seed
)
from .evaluation_utils import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    calculate_all_metrics,
    get_classification_report,
    get_confusion_matrix,
    print_evaluation_report,
    convert_predictions
)

__all__ = [
    # Original pipeline
    "DataPipeline",
    "DataInspector",
    "SwahiliTextCleaner",
    "load_datasets",
    "encode_labels",
    # Tokenizer utilities
    "load_tokenizer",
    "tokenize_text",
    "create_attention_mask",
    "get_tokenizer_info",
    "explain_special_tokens",
    # Transformer dataset
    "TransformerDataset",
    "create_data_loader",
    # Transformer preprocessing
    "prepare_transformer_data",
    "prepare_from_existing_datasets",
    # Configuration
    "TrainingConfig",
    # Device utilities
    "get_device",
    "move_to_device",
    "get_device_info",
    "print_device_info",
    "clear_gpu_cache",
    "set_seed",
    # Evaluation utilities
    "calculate_accuracy",
    "calculate_f1_score",
    "calculate_precision",
    "calculate_recall",
    "calculate_all_metrics",
    "get_classification_report",
    "get_confusion_matrix",
    "print_evaluation_report",
    "convert_predictions",
]

