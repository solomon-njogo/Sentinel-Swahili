"""
Llama 2 Tokenization Module for Swahili Text Data
Handles tokenization of preprocessed text using Llama 2 tokenizer for CLM fine-tuning.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
from transformers import AutoTokenizer
from datasets import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_llama_tokenizer(model_name: str = "meta-llama/Llama-2-7b-hf") -> Optional[AutoTokenizer]:
    """
    Load Llama 2 tokenizer with authentication and local path handling.
    Checks local paths first, then fetches from Hugging Face if not found locally.
    
    Args:
        model_name: Name of the Llama 2 model (default: "meta-llama/Llama-2-7b-hf")
        
    Returns:
        Tokenizer object or None if loading fails
    """
    logger.info(f"Attempting to load Llama 2 tokenizer: {model_name}")
    
    # First, try common local paths (faster if available)
    local_llama_paths = [
        "./models/llama-2-7b-hf",
        "./models/llama2-7b",
        os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf"),
    ]
    
    logger.info("Checking for local tokenizer files...")
    for local_path in local_llama_paths:
        expanded_path = os.path.expanduser(local_path)
        if os.path.exists(expanded_path):
            try:
                logger.info(f"Found local tokenizer at: {expanded_path}")
                logger.info("Attempting to load local tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    expanded_path,
                    use_fast=True,
                    local_files_only=True
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info("Set pad_token to eos_token for Llama 2 tokenizer")
                
                logger.info(f"Successfully loaded local Llama 2 tokenizer from: {expanded_path}")
                return tokenizer
                
            except Exception as e2:
                logger.debug(f"Failed to load tokenizer from {expanded_path}: {e2}")
                continue
    
    # If not found locally, fetch from Hugging Face Hub
    logger.info("Local tokenizer not found. Fetching from Hugging Face Hub...")
    logger.info("Note: This requires Hugging Face authentication for gated models.")
    logger.info("Run 'huggingface-cli login' if you haven't already.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=False
        )
        
        # Set padding token if not set (Llama 2 doesn't have a pad token by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token for Llama 2 tokenizer")
        
        logger.info(f"Successfully fetched and loaded Llama 2 tokenizer from Hugging Face: {model_name}")
        logger.info("Tokenizer has been cached locally for future use.")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to fetch tokenizer from Hugging Face Hub: {e}")
        logger.error("Please ensure you have:")
        logger.error("  1. Hugging Face authentication set up (huggingface-cli login)")
        logger.error("  2. Access to the Llama 2 model repository")
        logger.error("  3. Or a local copy of the tokenizer files")
        return None


def chunk_sequences(
    token_ids: List[int],
    block_size: int = 512,
    stride: int = 256
) -> List[List[int]]:
    """
    Chunk a sequence of token IDs into blocks of specified size with overlap.
    
    Args:
        token_ids: List of token IDs to chunk
        block_size: Size of each chunk (default: 512)
        stride: Stride between chunks for overlap (default: 256)
        
    Returns:
        List of token ID chunks
    """
    if len(token_ids) <= block_size:
        return [token_ids]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(token_ids):
        end_idx = start_idx + block_size
        chunk = token_ids[start_idx:end_idx]
        
        if len(chunk) == block_size:  # Only add full chunks
            chunks.append(chunk)
        
        start_idx += stride
        
        # If we've processed all tokens, break
        if start_idx >= len(token_ids):
            break
    
    return chunks


def tokenize_text_file(
    file_path: str,
    tokenizer: AutoTokenizer,
    block_size: int = 512,
    stride: int = 256,
    batch_size: int = 1000
) -> List[Dict[str, List[int]]]:
    """
    Tokenize a text file and chunk into sequences.
    
    Args:
        file_path: Path to the text file (one line per document)
        tokenizer: Tokenizer instance
        block_size: Size of each chunk in tokens (default: 512)
        stride: Stride between chunks for overlap (default: 256)
        batch_size: Number of lines to process at once (default: 1000)
        
    Returns:
        List of dictionaries with 'input_ids' key containing token chunks
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Tokenizing file: {file_path}")
    
    all_chunks = []
    total_lines = 0
    total_tokens = 0
    total_chunks = 0
    
    # Read and process file in batches
    with open(file_path, 'r', encoding='utf-8') as f:
        batch = []
        batch_line_count = 0
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            batch.append(line)
            batch_line_count += 1
            total_lines += 1
            
            # Process batch when it reaches batch_size
            if batch_line_count >= batch_size:
                try:
                    # Tokenize batch
                    encoded = tokenizer(
                        batch,
                        add_special_tokens=True,
                        truncation=False,
                        return_attention_mask=False
                    )
                    
                    # Chunk each sequence
                    for token_ids in encoded['input_ids']:
                        total_tokens += len(token_ids)
                        chunks = chunk_sequences(token_ids, block_size=block_size, stride=stride)
                        
                        for chunk in chunks:
                            all_chunks.append({'input_ids': chunk})
                            total_chunks += 1
                    
                    batch = []
                    batch_line_count = 0
                    
                except Exception as e:
                    logger.warning(f"Error tokenizing batch: {e}")
                    batch = []
                    batch_line_count = 0
                    continue
        
        # Process remaining batch
        if batch:
            try:
                encoded = tokenizer(
                    batch,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False
                )
                
                for token_ids in encoded['input_ids']:
                    total_tokens += len(token_ids)
                    chunks = chunk_sequences(token_ids, block_size=block_size, stride=stride)
                    
                    for chunk in chunks:
                        all_chunks.append({'input_ids': chunk})
                        total_chunks += 1
                        
            except Exception as e:
                logger.warning(f"Error tokenizing final batch: {e}")
    
    logger.info(f"Tokenization complete for {file_path}:")
    logger.info(f"  Total lines processed: {total_lines:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Total chunks created: {total_chunks:,}")
    logger.info(f"  Average tokens per line: {total_tokens / total_lines:.1f}" if total_lines > 0 else "  Average tokens per line: 0")
    logger.info(f"  Average tokens per chunk: {total_tokens / total_chunks:.1f}" if total_chunks > 0 else "  Average tokens per chunk: 0")
    
    return all_chunks


def tokenize_dataset(
    data_dir: str = "data",
    preprocessed_dir: str = "data/preprocessed",
    output_dir: str = "data/tokenized",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    block_size: int = 512,
    stride: int = 256,
    batch_size: int = 1000
) -> Dict[str, bool]:
    """
    Tokenize preprocessed dataset files using Llama 2 tokenizer.
    
    Args:
        data_dir: Base data directory (default: "data")
        preprocessed_dir: Directory containing preprocessed files (default: "data/preprocessed")
        output_dir: Directory to save tokenized datasets (default: "data/tokenized")
        model_name: Llama 2 model name (default: "meta-llama/Llama-2-7b-hf")
        block_size: Size of each chunk in tokens (default: 512)
        stride: Stride between chunks for overlap (default: 256)
        batch_size: Number of lines to process at once (default: 1000)
        
    Returns:
        Dictionary with success status for each dataset file
    """
    logger.info("=" * 80)
    logger.info("Starting Llama 2 Tokenization Pipeline")
    logger.info("=" * 80)
    
    # Log Swahili compatibility warning
    logger.warning("Note: Llama 2 tokenizer is primarily trained on English.")
    logger.warning("Swahili text may be tokenized less efficiently (more tokens per word).")
    logger.warning("This is expected and acceptable for domain adaptation fine-tuning.")
    
    # Load tokenizer
    tokenizer = load_llama_tokenizer(model_name=model_name)
    if tokenizer is None:
        logger.error("Failed to load Llama 2 tokenizer. Tokenization aborted.")
        return {
            'train': False,
            'test': False,
            'valid': False
        }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Tokenized datasets will be saved to: {output_path}")
    
    # Dataset files to process
    dataset_files = ["train.txt", "test.txt", "valid.txt"]
    results = {}
    
    preprocessed_path = Path(preprocessed_dir)
    
    for filename in dataset_files:
        input_path = preprocessed_path / filename
        
        if not input_path.exists():
            logger.warning(f"Preprocessed file not found, skipping: {input_path}")
            results[filename.replace('.txt', '')] = False
            continue
        
        try:
            logger.info(f"\nProcessing {filename}...")
            
            # Tokenize file
            chunks = tokenize_text_file(
                file_path=str(input_path),
                tokenizer=tokenizer,
                block_size=block_size,
                stride=stride,
                batch_size=batch_size
            )
            
            if not chunks:
                logger.warning(f"No chunks created for {filename}")
                results[filename.replace('.txt', '')] = False
                continue
            
            # Create Hugging Face dataset
            dataset = Dataset.from_list(chunks)
            
            # Save dataset
            output_file = output_path / filename.replace('.txt', '')
            dataset.save_to_disk(str(output_file))
            
            logger.info(f"Saved tokenized dataset to: {output_file}")
            logger.info(f"  Total examples: {len(dataset):,}")
            logger.info(f"  Features: {list(dataset.features.keys())}")
            
            results[filename.replace('.txt', '')] = True
            
        except Exception as e:
            logger.error(f"Error tokenizing {filename}: {e}", exc_info=True)
            results[filename.replace('.txt', '')] = False
            continue
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Tokenization Summary:")
    logger.info("=" * 80)
    for dataset_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {dataset_name}: {status}")
    
    logger.info("=" * 80)
    logger.info("Tokenization pipeline complete!")
    
    return results

