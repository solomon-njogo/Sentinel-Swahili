"""
Dataset Splitter Module for Swahili Text Data
Handles feature-target separation and label encoding for ML workflows.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from .data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_labels(labels: List[Optional[str]]) -> Tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
    """
    Convert string labels to integer encoding.
    
    Args:
        labels: List of labels (may contain None values)
        
    Returns:
        Tuple of (encoded_labels_array, label_mapping_dict)
        Returns (None, None) if no labels are present
    """
    # Filter out None values to check if we have any labels
    labeled = [l for l in labels if l is not None]
    
    if not labeled:
        logger.info("No labels found - returning None for encoded labels")
        return None, None
    
    # Create label mapping (string -> int)
    unique_labels = sorted(set(labeled))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    
    # Encode labels
    encoded = []
    for label in labels:
        if label is None:
            # Handle None values - use -1 as placeholder for missing labels
            encoded.append(-1)
        else:
            encoded.append(label_to_int[label])
    
    encoded_array = np.array(encoded, dtype=np.int64)
    
    logger.info(f"Encoded {len(unique_labels)} unique labels to integers")
    logger.info(f"Label mapping: {int_to_label}")
    
    return encoded_array, int_to_label


def load_datasets(data_dir: str = 'data') -> Dict[str, Any]:
    """
    Load and structure datasets into separate feature and target arrays.
    
    Args:
        data_dir: Directory containing data files (default: 'data')
        
    Returns:
        Dictionary containing:
            - X_train, X_test, X_valid: NumPy arrays of features (object dtype for text)
            - Y_train, Y_test, Y_valid: NumPy arrays of encoded labels (int64 dtype)
            - label_mapping: Dictionary mapping integer codes to original string labels
            - dataset_info: Dictionary with dataset statistics
    """
    logger.info(f"Loading datasets from {data_dir}")
    
    # Initialize pipeline
    pipeline = DataPipeline(data_dir=data_dir)
    
    # Dictionary to store results
    datasets = {}
    
    # Process each dataset file
    dataset_files = {
        'train': 'train.txt',
        'test': 'test.txt',
        'valid': 'valid.txt'
    }
    
    all_labels = []  # Collect all labels for consistent encoding
    
    # First pass: parse all datasets to collect all unique labels
    parsed_data = {}
    for split_name, filename in dataset_files.items():
        try:
            logger.info(f"Parsing {filename}...")
            features, labels, label_format = pipeline.parse_dataset(filename)
            
            if not features:
                logger.warning(f"No features found in {filename}")
                parsed_data[split_name] = {
                    'features': [],
                    'labels': [],
                    'label_format': None
                }
                continue
            
            # Validate feature-label alignment
            if labels and len(features) != len(labels):
                raise ValueError(
                    f"Mismatch in {filename}: {len(features)} features but {len(labels)} labels"
                )
            
            parsed_data[split_name] = {
                'features': features,
                'labels': labels,
                'label_format': label_format
            }
            
            # Collect labels for consistent encoding across all splits
            if labels:
                all_labels.extend([l for l in labels if l is not None])
            
            logger.info(f"Successfully parsed {len(features)} samples from {filename}")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            parsed_data[split_name] = {
                'features': [],
                'labels': [],
                'label_format': None
            }
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
    
    # Second pass: encode labels and convert to NumPy arrays
    for split_name, data in parsed_data.items():
        features = data['features']
        labels = data['labels']
        
        # Convert features to NumPy array (object dtype for strings)
        if features:
            X_array = np.array(features, dtype=object)
            datasets[f'X_{split_name}'] = X_array
            logger.info(f"Created X_{split_name} with shape {X_array.shape}")
        else:
            datasets[f'X_{split_name}'] = np.array([], dtype=object)
            logger.warning(f"X_{split_name} is empty")
        
        # Encode labels if they exist
        if labels and label_mapping is not None:
            # Use the unified label mapping
            label_to_int = {label: idx for idx, label in label_mapping.items()}
            encoded = []
            for label in labels:
                if label is None:
                    encoded.append(-1)  # Use -1 for missing labels
                else:
                    encoded.append(label_to_int[label])
            
            Y_array = np.array(encoded, dtype=np.int64)
            datasets[f'Y_{split_name}'] = Y_array
            logger.info(f"Created Y_{split_name} with shape {Y_array.shape}")
            
            # Log label distribution
            unique_encoded = np.unique(Y_array[Y_array >= 0])  # Exclude -1 (missing labels)
            logger.info(f"Y_{split_name} contains {len(unique_encoded)} unique classes")
        else:
            datasets[f'Y_{split_name}'] = None
            logger.info(f"Y_{split_name} is None (no labels)")
    
    # Add label mapping to results
    datasets['label_mapping'] = label_mapping
    
    # Add dataset information
    dataset_info = {}
    for split_name, data in parsed_data.items():
        dataset_info[split_name] = {
            'num_samples': len(data['features']),
            'has_labels': data['labels'] is not None and any(l is not None for l in data['labels']),
            'label_format': data['label_format']
        }
    
    datasets['dataset_info'] = dataset_info
    
    logger.info("Dataset loading complete!")
    logger.info(f"Train samples: {len(parsed_data['train']['features'])}")
    logger.info(f"Test samples: {len(parsed_data['test']['features'])}")
    logger.info(f"Valid samples: {len(parsed_data['valid']['features'])}")
    
    return datasets


if __name__ == "__main__":
    # Test the dataset splitter
    # Note: Run from project root: python -m src.dataset_splitter
    import sys
    from pathlib import Path
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    print("=" * 80)
    print("Testing Dataset Splitter")
    print("=" * 80)
    
    try:
        datasets = load_datasets(data_dir='data')
        
        print("\nDataset Structure:")
        print(f"  X_train shape: {datasets['X_train'].shape if datasets['X_train'] is not None else 'None'}")
        print(f"  X_test shape: {datasets['X_test'].shape if datasets['X_test'] is not None else 'None'}")
        print(f"  X_valid shape: {datasets['X_valid'].shape if datasets['X_valid'] is not None else 'None'}")
        
        print(f"\n  Y_train shape: {datasets['Y_train'].shape if datasets['Y_train'] is not None else 'None'}")
        print(f"  Y_test shape: {datasets['Y_test'].shape if datasets['Y_test'] is not None else 'None'}")
        print(f"  Y_valid shape: {datasets['Y_valid'].shape if datasets['Y_valid'] is not None else 'None'}")
        
        if datasets['label_mapping']:
            print(f"\n  Label mapping: {datasets['label_mapping']}")
        else:
            print("\n  No label mapping (unlabeled dataset)")
        
        print("\nDataset Info:")
        for split_name, info in datasets['dataset_info'].items():
            print(f"  {split_name}: {info}")
        
        # Show sample data
        if len(datasets['X_train']) > 0:
            print("\nSample Training Data:")
            print(f"  X_train[0]: {datasets['X_train'][0][:100]}...")
            if datasets['Y_train'] is not None:
                print(f"  Y_train[0]: {datasets['Y_train'][0]}")
                if datasets['label_mapping']:
                    label = datasets['label_mapping'].get(int(datasets['Y_train'][0]), 'Unknown')
                    print(f"  Decoded label: {label}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

