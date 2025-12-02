"""
Exploratory Data Analysis (EDA) Module for Swahili Dataset
Performs comprehensive analysis including vocabulary, tokenization, language distribution, and topic modeling.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_texts_from_file(file_path: str) -> List[str]:
    """
    Load text lines from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of text lines (non-empty)
    """
    texts = []
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return texts
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Only include non-empty lines
                texts.append(line)
    
    logger.info(f"Loaded {len(texts):,} lines from {file_path}")
    return texts


def analyze_vocabulary(texts: List[str]) -> Dict:
    """
    Analyze vocabulary: word frequency, unique words, vocabulary size.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary containing vocabulary statistics
    """
    logger.info("Analyzing vocabulary...")
    
    # Tokenize all texts into words
    all_words = []
    for text in texts:
        # Simple word tokenization (split on whitespace and punctuation)
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    # Calculate word frequencies
    word_freq = Counter(all_words)
    
    # Calculate statistics
    total_words = len(all_words)
    unique_words = len(word_freq)
    
    # Get top words
    top_words = dict(word_freq.most_common(50))
    
    vocabulary_stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'word_frequencies': dict(word_freq),
        'top_words': top_words,
        'average_words_per_document': total_words / len(texts) if texts else 0
    }
    
    logger.info(f"Vocabulary analysis complete: {unique_words:,} unique words from {total_words:,} total words")
    return vocabulary_stats


def load_tokenizer(model_name: Optional[str] = None) -> Optional:
    """
    Load a tokenizer for token counting.
    Uses GPT-2 tokenizer by default (accessible without authentication).
    Optionally tries Llama 2 tokenizer if model_name is provided or if available locally.
    
    Note: GPT-2 tokenizer provides approximate token counts. For exact Llama 2
    tokenization, you need access to the gated Llama 2 model repository or a local copy.
    
    Args:
        model_name: Optional name of the model to load tokenizer from.
                    If None, tries local Llama 2 tokenizer, then falls back to GPT-2.
        
    Returns:
        Tokenizer object or None if loading fails
    """
    from transformers import AutoTokenizer, GPT2Tokenizer
    import os
    
    # If specific model requested, try it first
    if model_name:
        try:
            logger.info(f"Attempting to load requested tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            logger.info(f"Tokenizer {model_name} loaded successfully")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load {model_name} tokenizer: {e}")
            logger.info("Falling back to alternatives...")
    
    # Try local Llama 2 tokenizer if available (common paths)
    local_llama_paths = [
        "./models/llama-2-7b-hf",
        "./models/llama2-7b",
        "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf",
    ]
    
    for local_path in local_llama_paths:
        expanded_path = os.path.expanduser(local_path)
        if os.path.exists(expanded_path):
            try:
                logger.info(f"Attempting to load local Llama 2 tokenizer from: {expanded_path}")
                tokenizer = AutoTokenizer.from_pretrained(expanded_path, use_fast=True, local_files_only=True)
                logger.info("Local Llama 2 tokenizer loaded successfully")
                return tokenizer
            except Exception as e:
                logger.debug(f"Failed to load tokenizer from {expanded_path}: {e}")
                continue
    
    # Default to GPT-2 tokenizer (always accessible, no authentication required)
    try:
        logger.info("Loading GPT-2 tokenizer (default, no authentication required)...")
        logger.info("Note: GPT-2 tokenization provides approximate token counts.")
        logger.info("For exact Llama 2 counts, ensure Llama 2 tokenizer is available locally or with authentication.")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logger.info("GPT-2 tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load GPT-2 tokenizer: {e}")
        return None


def analyze_tokens(texts: List[str], tokenizer=None) -> Dict:
    """
    Analyze token counts and sequence length distribution.
    
    Args:
        texts: List of text strings
        tokenizer: Optional tokenizer object. If None, attempts to load one.
        
    Returns:
        Dictionary containing token statistics
    """
    logger.info("Analyzing tokens...")
    
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = load_tokenizer()
    
    if tokenizer is None:
        logger.warning("No tokenizer available, skipping token analysis")
        return {
            'tokenizer_available': False,
            'total_tokens': 0,
            'token_lengths': [],
            'average_tokens_per_document': 0,
            'min_tokens': 0,
            'max_tokens': 0,
            'median_tokens': 0
        }
    
    token_lengths = []
    total_tokens = 0
    
    # Process texts in batches for efficiency
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Tokenize batch
            encoded = tokenizer(batch, add_special_tokens=False, return_length=True)
            
            # Extract token lengths
            if isinstance(encoded, dict) and 'length' in encoded:
                batch_lengths = encoded['length']
            else:
                # Fallback: tokenize individually
                batch_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in batch]
            
            token_lengths.extend(batch_lengths)
            total_tokens += sum(batch_lengths)
            
        except Exception as e:
            logger.warning(f"Error tokenizing batch {i//batch_size + 1}: {e}")
            # Fallback to individual tokenization
            for text in batch:
                try:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    length = len(tokens)
                    token_lengths.append(length)
                    total_tokens += length
                except Exception as e2:
                    logger.warning(f"Error tokenizing individual text: {e2}")
                    token_lengths.append(0)
    
    # Calculate statistics
    if token_lengths:
        sorted_lengths = sorted(token_lengths)
        median_idx = len(sorted_lengths) // 2
        median_tokens = sorted_lengths[median_idx] if sorted_lengths else 0
    else:
        median_tokens = 0
    
    token_stats = {
        'tokenizer_available': True,
        'total_tokens': total_tokens,
        'token_lengths': token_lengths,
        'average_tokens_per_document': total_tokens / len(texts) if texts else 0,
        'min_tokens': min(token_lengths) if token_lengths else 0,
        'max_tokens': max(token_lengths) if token_lengths else 0,
        'median_tokens': median_tokens,
        'documents_over_4096_tokens': sum(1 for length in token_lengths if length > 4096),
        'documents_over_2048_tokens': sum(1 for length in token_lengths if length > 2048),
        'documents_over_1024_tokens': sum(1 for length in token_lengths if length > 1024)
    }
    
    logger.info(f"Token analysis complete: {total_tokens:,} total tokens, "
                f"avg {token_stats['average_tokens_per_document']:.1f} tokens per document")
    return token_stats


def analyze_language_distribution(texts: List[str]) -> Dict:
    """
    Analyze language distribution using langdetect.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary containing language distribution statistics
    """
    logger.info("Analyzing language distribution...")
    
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        logger.warning("langdetect not available, skipping language analysis")
        return {
            'languages_detected': {},
            'swahili_percentage': 0.0,
            'total_detected': 0,
            'detection_errors': len(texts)
        }
    
    language_counts = Counter()
    detection_errors = 0
    
    # Sample texts if dataset is very large (for efficiency)
    sample_size = min(10000, len(texts))
    if len(texts) > sample_size:
        import random
        sample_texts = random.sample(texts, sample_size)
        logger.info(f"Sampling {sample_size:,} texts from {len(texts):,} for language detection")
    else:
        sample_texts = texts
    
    for text in sample_texts:
        try:
            # Skip very short texts (langdetect needs sufficient content)
            if len(text.strip()) < 10:
                continue
            
            detected_lang = detect(text)
            language_counts[detected_lang] += 1
        except LangDetectException:
            detection_errors += 1
        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            detection_errors += 1
    
    total_detected = sum(language_counts.values())
    swahili_count = language_counts.get('sw', 0)
    swahili_percentage = (swahili_count / total_detected * 100) if total_detected > 0 else 0.0
    
    language_stats = {
        'languages_detected': dict(language_counts),
        'swahili_percentage': swahili_percentage,
        'swahili_count': swahili_count,
        'total_detected': total_detected,
        'detection_errors': detection_errors,
        'sample_size': len(sample_texts)
    }
    
    logger.info(f"Language analysis complete: {swahili_percentage:.1f}% Swahili "
                f"({swahili_count}/{total_detected} detected)")
    return language_stats


def analyze_topics(texts: List[str], num_topics: int = 10, num_words: int = 10) -> Dict:
    """
    Perform LDA topic modeling on the texts.
    
    Args:
        texts: List of text strings
        num_topics: Number of topics to extract
        num_words: Number of top words per topic to return
        
    Returns:
        Dictionary containing topic modeling results
    """
    logger.info(f"Performing topic modeling with {num_topics} topics...")
    
    try:
        from gensim import corpora
        from gensim.models import LdaModel
        from gensim.utils import simple_preprocess
    except ImportError:
        logger.warning("gensim not available, skipping topic modeling")
        return {
            'topics': [],
            'model_available': False
        }
    
    # Sample texts if dataset is very large (LDA can be slow on large datasets)
    sample_size = min(5000, len(texts))
    if len(texts) > sample_size:
        import random
        sample_texts = random.sample(texts, sample_size)
        logger.info(f"Sampling {sample_size:,} texts from {len(texts):,} for topic modeling")
    else:
        sample_texts = texts
    
    try:
        # Preprocess texts: tokenize and create dictionary
        processed_texts = [simple_preprocess(text, deacc=True, min_len=2) for text in sample_texts]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        
        # Filter extremes (remove very common and very rare words)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        # Create corpus
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        logger.info("Training LDA model...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics
        topics = []
        for topic_id in range(num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=num_words)
            topics.append({
                'topic_id': topic_id,
                'words': [{'word': word, 'weight': weight} for word, weight in topic_words]
            })
        
        topic_stats = {
            'topics': topics,
            'model_available': True,
            'num_topics': num_topics,
            'vocabulary_size': len(dictionary),
            'sample_size': len(sample_texts)
        }
        
        logger.info(f"Topic modeling complete: {num_topics} topics extracted")
        return topic_stats
        
    except Exception as e:
        logger.error(f"Error in topic modeling: {e}")
        return {
            'topics': [],
            'model_available': False,
            'error': str(e)
        }


def run_eda(data_dir: str, output_dir: str) -> Dict:
    """
    Run complete EDA analysis on preprocessed dataset files.
    
    Args:
        data_dir: Directory containing preprocessed files (should contain 'preprocessed' subdirectory)
        output_dir: Directory to save EDA results and reports
        
    Returns:
        Dictionary containing all EDA results
    """
    logger.info("=" * 80)
    logger.info("Starting Exploratory Data Analysis (EDA)")
    logger.info("=" * 80)
    
    data_path = Path(data_dir)
    preprocessed_dir = data_path / "preprocessed"
    
    if not preprocessed_dir.exists():
        logger.error(f"Preprocessed directory not found: {preprocessed_dir}")
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"EDA results will be saved to: {output_path}")
    
    # Dataset files to analyze
    dataset_files = ["train.txt", "test.txt", "valid.txt"]
    
    all_results = {}
    all_texts = []
    
    # Load and analyze each file
    for filename in dataset_files:
        file_path = preprocessed_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            continue
        
        logger.info(f"\nAnalyzing {filename}...")
        texts = load_texts_from_file(str(file_path))
        
        if not texts:
            logger.warning(f"No texts found in {filename}, skipping analysis")
            continue
        
        all_texts.extend(texts)
        
        # Run analyses for this file
        file_results = {
            'filename': filename,
            'num_documents': len(texts),
            'vocabulary': analyze_vocabulary(texts),
            'tokens': analyze_tokens(texts),
            'language': analyze_language_distribution(texts),
            'topics': analyze_topics(texts)
        }
        
        all_results[filename] = file_results
    
    # Aggregate analysis across all files
    logger.info("\n" + "=" * 80)
    logger.info("Running aggregate analysis across all files...")
    logger.info("=" * 80)
    
    if all_texts:
        aggregate_results = {
            'total_documents': len(all_texts),
            'vocabulary': analyze_vocabulary(all_texts),
            'tokens': analyze_tokens(all_texts),
            'language': analyze_language_distribution(all_texts),
            'topics': analyze_topics(all_texts)
        }
        all_results['aggregate'] = aggregate_results
    
    logger.info("=" * 80)
    logger.info("EDA Analysis Complete!")
    logger.info("=" * 80)
    
    return all_results

