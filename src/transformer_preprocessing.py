"""
Data Preprocessing for Transformer Models
Prepares data from the existing pipeline for transformer fine-tuning.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from .data_pipeline import DataPipeline
from .dataset_splitter import load_datasets, encode_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_transformer_data(
    data_dir: str = 'data',
    use_raw_text: bool = True,
    files: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prepare data for transformer models from the existing data pipeline.
    
    Args:
        data_dir: Directory containing data files (default: 'data')
        use_raw_text: If True, use raw text without heavy cleaning (stemming/stopwords).
                     If False, use cleaned text from the pipeline (default: True)
        files: List of files to process. If None, uses ['train.txt', 'test.txt', 'valid.txt']
    
    Returns:
        Dictionary containing:
        - texts_train, texts_test, texts_valid: Lists of text strings
        - labels_train, labels_test, labels_valid: Lists of labels (or None if unlabeled)
        - label_mapping: Dictionary mapping integer labels to string labels
        - dataset_info: Dictionary with dataset statistics
    """
    if files is None:
        files = ['train.txt', 'test.txt', 'valid.txt']
    
    logger.info(f"Preparing transformer data from {data_dir}")
    logger.info(f"Using raw text: {use_raw_text}")
    
    # Initialize pipeline
    pipeline = DataPipeline(data_dir=data_dir)
    
    # Dictionary to store results
    result = {}
    
    # Process each dataset file
    dataset_files = {
        'train': 'train.txt',
        'test': 'test.txt',
        'valid': 'valid.txt'
    }
    
    all_labels = []  # Collect all labels for consistent encoding
    parsed_data = {}
    
    # First pass: parse all datasets
    for split_name, filename in dataset_files.items():
        if filename not in files:
            logger.warning(f"Skipping {filename} (not in files list)")
            parsed_data[split_name] = {'features': [], 'labels': [], 'label_format': None}
            continue
        
        try:
            logger.info(f"Processing {filename}...")
            
            if use_raw_text:
                # Get raw text without heavy cleaning
                # Read file directly and parse without applying text cleaning
                lines = pipeline.read_file(filename)
                label_format = pipeline.detect_label_format(lines)
                
                features = []
                labels = []
                for line in lines:
                    text, label = pipeline.parse_line(line, label_format)
                    # Only apply minimal cleaning: normalize encoding and remove noise
                    # Skip: lowercase, stemming, stopword removal, basic tokenization
                    # IMPORTANT: Preserve UNK tokens - they are part of the original data
                    # and should be passed to the transformer tokenizer as-is
                    text = pipeline.text_cleaner.normalize_encoding(text)
                    text = pipeline.text_cleaner.remove_noise(text)
                    # UNK tokens are preserved (not removed) for transformer compatibility
                    features.append(text)
                    labels.append(label)
            else:
                # Use existing pipeline with full cleaning
                features, labels, label_format = pipeline.parse_dataset(filename)
            
            if not features:
                logger.warning(f"No features found in {filename}")
                parsed_data[split_name] = {'features': [], 'labels': [], 'label_format': None}
                continue
            
            parsed_data[split_name] = {
                'features': features,
                'labels': labels,
                'label_format': label_format
            }
            
            # Collect labels
            if labels:
                all_labels.extend([l for l in labels if l is not None])
            
            logger.info(f"Processed {len(features)} samples from {filename}")
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            parsed_data[split_name] = {'features': [], 'labels': [], 'label_format': None}
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            raise
    
    # Create unified label encoding if labels exist
    label_mapping = None
    if all_labels:
        unique_labels = sorted(set(all_labels))
        label_mapping = {idx: label for idx, label in enumerate(unique_labels)}
        label_to_int = {label: idx for label, idx in label_mapping.items()}
        logger.info(f"Created unified label encoding with {len(unique_labels)} classes")
    else:
        logger.info("No labels found in any dataset - unlabeled data")
    
    # Extract texts and labels for each split
    for split_name, data in parsed_data.items():
        features = data['features']
        labels = data['labels']
        
        result[f'texts_{split_name}'] = features
        
        # Encode labels if they exist
        if labels and label_mapping is not None:
            label_to_int = {label: idx for idx, label in label_mapping.items()}
            encoded_labels = []
            for label in labels:
                if label is None:
                    encoded_labels.append(-1)  # Use -1 for missing labels
                else:
                    encoded_labels.append(label_to_int[label])
            result[f'labels_{split_name}'] = encoded_labels
        else:
            result[f'labels_{split_name}'] = None
    
    # Add metadata
    result['label_mapping'] = label_mapping
    
    # Add dataset information
    dataset_info = {}
    for split_name, data in parsed_data.items():
        dataset_info[split_name] = {
            'num_samples': len(data['features']),
            'has_labels': data['labels'] is not None and any(l is not None for l in data['labels']),
            'label_format': data['label_format']
        }
    
    result['dataset_info'] = dataset_info
    
    logger.info("Transformer data preparation complete!")
    logger.info(f"Train samples: {len(result['texts_train'])}")
    logger.info(f"Test samples: {len(result['texts_test'])}")
    logger.info(f"Valid samples: {len(result['texts_valid'])}")
    
    return result


def prepare_from_existing_datasets(
    datasets: Dict[str, Any],
    use_raw_text: bool = True
) -> Dict[str, Any]:
    """
    Prepare transformer data from existing dataset dictionary (from load_datasets).
    
    Args:
        datasets: Dictionary from load_datasets() containing X_train, Y_train, etc.
        use_raw_text: Whether to use raw text (not applicable here, kept for compatibility)
    
    Returns:
        Dictionary with texts and labels for transformer models
    """
    logger.info("Preparing transformer data from existing datasets")
    
    result = {}
    
    splits = ['train', 'test', 'valid']
    
    for split in splits:
        X_key = f'X_{split}'
        Y_key = f'Y_{split}'
        
        if X_key in datasets and datasets[X_key] is not None:
            # Convert numpy array to list of strings
            texts = datasets[X_key].tolist() if isinstance(datasets[X_key], np.ndarray) else list(datasets[X_key])
            result[f'texts_{split}'] = texts
        else:
            result[f'texts_{split}'] = []
        
        if Y_key in datasets and datasets[Y_key] is not None:
            # Convert numpy array to list
            labels = datasets[Y_key].tolist() if isinstance(datasets[Y_key], np.ndarray) else list(datasets[Y_key])
            # Filter out -1 (missing labels) if needed
            result[f'labels_{split}'] = labels
        else:
            result[f'labels_{split}'] = None
    
    # Copy label mapping and dataset info
    result['label_mapping'] = datasets.get('label_mapping', None)
    result['dataset_info'] = datasets.get('dataset_info', {})
    
    return result


if __name__ == "__main__":
    # Test transformer preprocessing
    print("=" * 80)
    print("Testing Transformer Preprocessing")
    print("=" * 80)
    
    # Test with raw text
    print("\nTesting with raw text (recommended for transformers):")
    data_raw = prepare_transformer_data(data_dir='data', use_raw_text=True)
    
    print(f"\nTrain samples: {len(data_raw['texts_train'])}")
    print(f"Test samples: {len(data_raw['texts_test'])}")
    print(f"Valid samples: {len(data_raw['texts_valid'])}")
    
    if data_raw['label_mapping']:
        print(f"\nLabel mapping: {data_raw['label_mapping']}")
        print(f"Train labels sample: {data_raw['labels_train'][:5] if data_raw['labels_train'] else None}")
    else:
        print("\nNo labels found")
    
    # Show sample text
    if data_raw['texts_train']:
        print(f"\nSample train text: {data_raw['texts_train'][0][:100]}...")
    
    print("\n" + "=" * 80)
    print("Transformer Preprocessing Test Complete!")
    print("=" * 80)

