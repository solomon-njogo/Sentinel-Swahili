"""
PyTorch Dataset Class for Transformer Models
Handles efficient data loading for transformer fine-tuning.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Union
import logging
import numpy as np
from transformers import AutoTokenizer
from .tokenizer_utils import load_tokenizer, tokenize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerDataset(Dataset):
    """
    PyTorch Dataset class for transformer models.
    Handles text and labels for supervised learning tasks.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[Union[int, str]]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_name: str = "xlm-roberta-base",
        max_length: int = 512,
        cache_tokenizer: bool = True
    ):
        """
        Initialize the transformer dataset.
        
        Args:
            texts: List of text strings
            labels: Optional list of labels (integers or strings). If None, dataset is unlabeled.
            tokenizer: Pre-loaded tokenizer instance (optional). If None, will load from tokenizer_name.
            tokenizer_name: Name of tokenizer to load if tokenizer is None (default: xlm-roberta-base)
            max_length: Maximum sequence length for tokenization (default: 512)
            cache_tokenizer: Whether to cache the tokenizer instance (default: True)
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Load or use provided tokenizer
        if tokenizer is None:
            self.tokenizer = load_tokenizer(tokenizer_name)
        else:
            self.tokenizer = tokenizer
        
        # Validate data
        if labels is not None and len(texts) != len(labels):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(labels)} labels"
            )
        
        # Check if labels are present
        self.has_labels = labels is not None
        
        # Convert string labels to integers if needed
        if self.has_labels and labels:
            # Check if labels are strings
            if isinstance(labels[0], str):
                unique_labels = sorted(set(labels))
                self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
                self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
                self.num_classes = len(unique_labels)
                logger.info(f"Converted {len(unique_labels)} string labels to integers")
            else:
                # Labels are already integers
                self.label_to_int = None
                self.int_to_label = None
                unique_labels = set(labels)
                self.num_classes = len(unique_labels)
        
        logger.info(f"Initialized TransformerDataset with {len(texts)} samples")
        logger.info(f"Max length: {max_length}, Has labels: {self.has_labels}")
        if self.has_labels:
            logger.info(f"Number of classes: {self.num_classes}")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
        
        Returns:
            Dictionary containing:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Label (if available)
            - text: Original text (for debugging)
        """
        text = self.texts[idx]
        
        # Tokenize text
        encoded = tokenize_text(
            self.tokenizer,
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract tensors (remove batch dimension since we're processing single items)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text  # Keep original text for debugging
        }
        
        # Add label if available
        if self.has_labels:
            label = self.labels[idx]
            
            # Convert string label to integer if needed
            if isinstance(label, str) and self.label_to_int is not None:
                label = self.label_to_int[label]
            
            result['labels'] = torch.tensor(label, dtype=torch.long)
        
        return result
    
    def get_label_mapping(self) -> Optional[Dict[int, str]]:
        """
        Get the mapping from integer labels to string labels.
        
        Returns:
            Dictionary mapping integer labels to string labels, or None if labels are already integers
        """
        return self.int_to_label
    
    def get_num_classes(self) -> Optional[int]:
        """
        Get the number of classes in the dataset.
        
        Returns:
            Number of classes, or None if dataset is unlabeled
        """
        return self.num_classes if self.has_labels else None


def create_data_loader(
    dataset: TransformerDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the transformer dataset.
    
    Args:
        dataset: TransformerDataset instance
        batch_size: Batch size (default: 16)
        shuffle: Whether to shuffle the data (default: True)
        num_workers: Number of worker processes for data loading (default: 0)
        pin_memory: Whether to pin memory for faster GPU transfer (default: False)
    
    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


if __name__ == "__main__":
    # Test the transformer dataset
    print("=" * 80)
    print("Testing TransformerDataset")
    print("=" * 80)
    
    # Create sample data
    sample_texts = [
        "Hii ni mfano wa maandishi ya Kiswahili.",
        "This is an example of Swahili text.",
        "Jambo la dunia!",
        "Habari za asubuhi?",
        "Karibu sana!"
    ]
    
    sample_labels = ["positive", "positive", "neutral", "positive", "positive"]
    
    # Create dataset
    dataset = TransformerDataset(
        texts=sample_texts,
        labels=sample_labels,
        max_length=128
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Has labels: {dataset.has_labels}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Label mapping: {dataset.get_label_mapping()}")
    
    # Get a sample
    print("\n" + "=" * 80)
    print("Sample Item")
    print("=" * 80)
    sample = dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label: {sample['labels']}")
    print(f"Text: {sample['text']}")
    
    # Test DataLoader
    print("\n" + "=" * 80)
    print("Testing DataLoader")
    print("=" * 80)
    data_loader = create_data_loader(dataset, batch_size=2, shuffle=False)
    
    for batch_idx, batch in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Labels: {batch['labels'].tolist()}")
        if batch_idx >= 1:  # Just show first 2 batches
            break
    
    # Test unlabeled dataset
    print("\n" + "=" * 80)
    print("Testing Unlabeled Dataset")
    print("=" * 80)
    unlabeled_dataset = TransformerDataset(
        texts=sample_texts,
        labels=None,
        max_length=128
    )
    
    print(f"Has labels: {unlabeled_dataset.has_labels}")
    sample_unlabeled = unlabeled_dataset[0]
    print(f"Sample keys: {sample_unlabeled.keys()}")
    print("'labels' in sample:", 'labels' in sample_unlabeled)
    
    print("\n" + "=" * 80)
    print("TransformerDataset Test Complete!")
    print("=" * 80)

