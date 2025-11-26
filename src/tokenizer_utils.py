"""
Tokenizer Utilities for Transformer Models
Handles loading and using tokenizers for RoBERTa and other transformer models.

Special Tokens:
- <s>: Beginning of sequence token (BOS)
- </s>: End of sequence token (EOS)
- <pad>: Padding token
- <unk> or UNK: Unknown token (may appear in your data as "UNK")
- <mask>: Mask token (for masked language modeling)

Note: Your Swahili data contains "UNK" tokens which should be preserved
and passed to the tokenizer as-is. The tokenizer will handle them appropriately.
"""

import logging
from typing import List, Dict, Optional, Union, Any
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name: str = "xlm-roberta-base",
    cache_dir: Optional[str] = None,
    use_fast: bool = True
) -> AutoTokenizer:
    """
    Load a tokenizer for a transformer model.
    
    Args:
        model_name: Name of the model/tokenizer to load (default: xlm-roberta-base)
                   Examples: 'xlm-roberta-base', 'roberta-base', 'bert-base-uncased'
        cache_dir: Directory to cache the tokenizer (optional)
        use_fast: Whether to use the fast tokenizer implementation (default: True)
    
    Returns:
        Loaded tokenizer instance
    """
    logger.info(f"Loading tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=use_fast
        )
        logger.info(f"Successfully loaded tokenizer: {model_name}")
        
        # Log tokenizer info
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")
        logger.info(f"Tokenizer cls token: {tokenizer.cls_token}")
        logger.info(f"Tokenizer sep token: {tokenizer.sep_token}")
        
        return tokenizer
    
    except Exception as e:
        logger.error(f"Error loading tokenizer {model_name}: {e}")
        raise


def tokenize_text(
    tokenizer: AutoTokenizer,
    text: Union[str, List[str]],
    max_length: int = 512,
    padding: Union[bool, str] = True,
    truncation: bool = True,
    return_tensors: Optional[str] = None,
    return_attention_mask: bool = True
) -> Dict[str, Any]:
    """
    Tokenize text(s) using the provided tokenizer.
    
    Args:
        tokenizer: The tokenizer instance to use
        text: Single text string or list of text strings
        max_length: Maximum sequence length (default: 512)
        padding: Padding strategy - True, False, 'max_length', or 'longest' (default: True)
        truncation: Whether to truncate sequences (default: True)
        return_tensors: Return format - 'pt' for PyTorch, 'tf' for TensorFlow, None for lists (default: None)
        return_attention_mask: Whether to return attention masks (default: True)
    
    Returns:
        Dictionary containing tokenized inputs:
        - input_ids: Token IDs
        - attention_mask: Attention masks (if return_attention_mask=True)
        - Additional fields depending on tokenizer
    """
    if isinstance(text, str):
        text = [text]
    
    logger.debug(f"Tokenizing {len(text)} text(s) with max_length={max_length}")
    
    # Tokenize
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
        return_attention_mask=return_attention_mask
    )
    
    return encoded


def create_attention_mask(input_ids: Any) -> Any:
    """
    Create attention mask from input IDs.
    Typically, 1 for real tokens and 0 for padding tokens.
    
    Args:
        input_ids: Token IDs (can be tensor, list, or array)
    
    Returns:
        Attention mask with same shape as input_ids
    """
    import torch
    import numpy as np
    
    # Convert to tensor if needed
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    elif isinstance(input_ids, np.ndarray):
        input_ids = torch.from_numpy(input_ids)
    
    # Create mask: 1 where input_ids != pad_token_id, 0 otherwise
    # This assumes pad_token_id is 1 (common for RoBERTa) or can be obtained from tokenizer
    # For a more robust solution, pass tokenizer.pad_token_id
    attention_mask = (input_ids != 0).long()
    
    return attention_mask


def get_tokenizer_info(tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Get information about a tokenizer.
    
    Args:
        tokenizer: The tokenizer instance
    
    Returns:
        Dictionary with tokenizer information
    """
    info = {
        "model_name": tokenizer.name_or_path,
        "vocab_size": len(tokenizer),
        "pad_token": str(tokenizer.pad_token),
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token": str(tokenizer.cls_token),
        "cls_token_id": tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None,
        "sep_token": str(tokenizer.sep_token),
        "sep_token_id": tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None,
        "unk_token": str(tokenizer.unk_token),
        "unk_token_id": tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None,
        "model_max_length": tokenizer.model_max_length,
        "is_fast": tokenizer.is_fast if hasattr(tokenizer, 'is_fast') else False,
    }
    
    # Add special tokens information
    info["special_tokens"] = {
        "bos_token": str(tokenizer.bos_token) if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token else None,
        "eos_token": str(tokenizer.eos_token) if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token else None,
        "mask_token": str(tokenizer.mask_token) if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token else None,
    }
    
    return info


def explain_special_tokens(tokenizer: AutoTokenizer) -> Dict[str, str]:
    """
    Explain the special tokens used by the tokenizer.
    
    Args:
        tokenizer: The tokenizer instance
    
    Returns:
        Dictionary explaining each special token
    """
    explanations = {}
    
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
        explanations["BOS"] = {
            "token": str(tokenizer.bos_token),
            "name": "Beginning of Sequence",
            "purpose": "Marks the start of the input sequence"
        }
    
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        explanations["EOS"] = {
            "token": str(tokenizer.eos_token),
            "name": "End of Sequence",
            "purpose": "Marks the end of the input sequence"
        }
    
    if tokenizer.pad_token:
        explanations["PAD"] = {
            "token": str(tokenizer.pad_token),
            "name": "Padding",
            "purpose": "Used to pad sequences to the same length in batches"
        }
    
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
        explanations["UNK"] = {
            "token": str(tokenizer.unk_token),
            "name": "Unknown",
            "purpose": "Represents tokens not in the vocabulary. Note: Your data may contain 'UNK' as a string, which should be preserved."
        }
    
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
        explanations["MASK"] = {
            "token": str(tokenizer.mask_token),
            "name": "Mask",
            "purpose": "Used in masked language modeling tasks"
        }
    
    return explanations


if __name__ == "__main__":
    # Test tokenizer utilities
    print("=" * 80)
    print("Testing Tokenizer Utilities")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = load_tokenizer("xlm-roberta-base")
    
    # Get tokenizer info
    info = get_tokenizer_info(tokenizer)
    print("\nTokenizer Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test tokenization
    test_texts = [
        "Hii ni mfano wa maandishi ya Kiswahili.",
        "This is an example of Swahili text.",
        "Jambo la dunia!"
    ]
    
    print("\n" + "=" * 80)
    print("Testing Tokenization")
    print("=" * 80)
    
    for text in test_texts:
        encoded = tokenize_text(
            tokenizer,
            text,
            max_length=128,
            padding=True,
            truncation=True
        )
        
        print(f"\nOriginal text: {text}")
        print(f"Token IDs length: {len(encoded['input_ids'][0])}")
        print(f"Attention mask length: {len(encoded['attention_mask'][0])}")
        print(f"First 10 token IDs: {encoded['input_ids'][0][:10]}")
        
        # Decode to verify
        decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)
        print(f"Decoded (with special tokens): {decoded[:100]}...")
    
    # Test batch tokenization
    print("\n" + "=" * 80)
    print("Testing Batch Tokenization")
    print("=" * 80)
    
    batch_encoded = tokenize_text(
        tokenizer,
        test_texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"Batch input_ids shape: {batch_encoded['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch_encoded['attention_mask'].shape}")
    
    print("\n" + "=" * 80)
    print("Tokenizer Utilities Test Complete!")
    print("=" * 80)

