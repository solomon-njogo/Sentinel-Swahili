"""
Visualization Module for EDA Results
Generates plots and charts for exploratory data analysis.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_word_frequency(word_freq: Dict[str, int], top_n: int, output_path: str, title: str = "Top Words") -> None:
    """
    Generate word frequency visualization (bar chart).
    
    Args:
        word_freq: Dictionary mapping words to frequencies
        top_n: Number of top words to display
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info(f"Generating word frequency plot (top {top_n} words)...")
    
    # Get top N words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [w[0] for w in sorted_words]
    frequencies = [w[1] for w in sorted_words]
    
    # Create plot
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(words)), frequencies, color='steelblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.title(f'{title} - Top {top_n} Most Frequent Words', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest frequency at top
    
    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        plt.text(freq, i, f' {freq:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Word frequency plot saved to: {output_path}")


def plot_token_distribution(token_lengths: List[int], output_path: str, title: str = "Token Distribution") -> None:
    """
    Plot sequence length distribution (histogram and box plot).
    
    Args:
        token_lengths: List of token counts per document
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info("Generating token distribution plot...")
    
    if not token_lengths:
        logger.warning("No token lengths available for plotting")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    axes[0].hist(token_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(1024, color='red', linestyle='--', linewidth=2, label='1024 tokens')
    axes[0].axvline(2048, color='orange', linestyle='--', linewidth=2, label='2048 tokens')
    axes[0].axvline(4096, color='purple', linestyle='--', linewidth=2, label='4096 tokens (Llama 2)')
    axes[0].set_xlabel('Sequence Length (tokens)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{title} - Histogram', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(token_lengths, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[1].axhline(1024, color='red', linestyle='--', linewidth=2, label='1024 tokens')
    axes[1].axhline(2048, color='orange', linestyle='--', linewidth=2, label='2048 tokens')
    axes[1].axhline(4096, color='purple', linestyle='--', linewidth=2, label='4096 tokens (Llama 2)')
    axes[1].set_ylabel('Sequence Length (tokens)', fontsize=12)
    axes[1].set_title(f'{title} - Box Plot', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics text
    import numpy as np
    stats_text = (
        f'Mean: {np.mean(token_lengths):.1f}\n'
        f'Median: {np.median(token_lengths):.1f}\n'
        f'Min: {np.min(token_lengths)}\n'
        f'Max: {np.max(token_lengths)}\n'
        f'Std: {np.std(token_lengths):.1f}'
    )
    axes[1].text(1.15, 0.5, stats_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Token distribution plot saved to: {output_path}")


def plot_language_distribution(lang_dist: Dict[str, int], output_path: str, title: str = "Language Distribution") -> None:
    """
    Visualize language distribution (pie chart and bar chart).
    
    Args:
        lang_dist: Dictionary mapping language codes to counts
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info("Generating language distribution plot...")
    
    if not lang_dist:
        logger.warning("No language distribution data available for plotting")
        return
    
    # Sort languages by count
    sorted_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
    languages = [lang[0] for lang in sorted_langs]
    counts = [lang[1] for lang in sorted_langs]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart (top 10 languages)
    top_n = min(10, len(languages))
    top_languages = languages[:top_n]
    top_counts = counts[:top_n]
    other_count = sum(counts[top_n:]) if len(languages) > top_n else 0
    
    if other_count > 0:
        top_languages.append('Other')
        top_counts.append(other_count)
    
    colors = plt.cm.Set3(range(len(top_languages)))
    axes[0].pie(top_counts, labels=top_languages, autopct='%1.1f%%',
                startangle=90, colors=colors)
    axes[0].set_title(f'{title} - Pie Chart (Top {top_n})', fontsize=13, fontweight='bold')
    
    # Bar chart
    bar_colors = ['green' if lang == 'sw' else 'steelblue' for lang in languages[:15]]
    axes[1].barh(range(min(15, len(languages))), counts[:15], color=bar_colors)
    axes[1].set_yticks(range(min(15, len(languages))))
    axes[1].set_yticklabels(languages[:15])
    axes[1].set_xlabel('Count', fontsize=12)
    axes[1].set_ylabel('Language Code', fontsize=12)
    axes[1].set_title(f'{title} - Bar Chart (Top 15)', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, count in enumerate(counts[:15]):
        axes[1].text(count, i, f' {count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Language distribution plot saved to: {output_path}")


def plot_topics(topics: List[Dict], num_topics: int, output_path: str, title: str = "Topic Modeling") -> None:
    """
    Visualize LDA topics (bar charts for top words per topic).
    
    Args:
        topics: List of topic dictionaries with 'topic_id' and 'words' (list of {word, weight})
        num_topics: Number of topics to display
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info(f"Generating topic visualization for {num_topics} topics...")
    
    if not topics:
        logger.warning("No topics available for plotting")
        return
    
    # Determine grid size
    import math
    cols = 3
    rows = math.ceil(min(num_topics, len(topics)) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    # Handle axes array - flatten if 2D, convert to list if 1D
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes.flatten())
    else:
        axes = axes.flatten()
    
    for idx, topic in enumerate(topics[:num_topics]):
        if idx >= len(axes):
            break
        
        topic_id = topic.get('topic_id', idx)
        words_data = topic.get('words', [])
        
        if not words_data:
            continue
        
        # Extract words and weights
        words = [w['word'] for w in words_data]
        weights = [w['weight'] for w in words_data]
        
        # Create bar chart for this topic
        ax = axes[idx]
        bars = ax.barh(range(len(words)), weights, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Weight', fontsize=10)
        ax.set_title(f'Topic {topic_id}', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, weight in enumerate(weights):
            ax.text(weight, i, f' {weight:.3f}', va='center', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(topics[:num_topics]), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{title} - Top Words per Topic', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Topic visualization saved to: {output_path}")


def display_word_frequency_terminal(word_freq: Dict[str, int], top_n: int, title: str = "Top Words") -> None:
    """
    Display word frequency summary in terminal.
    
    Args:
        word_freq: Dictionary mapping words to frequencies
        top_n: Number of top words to display
        title: Display title
    """
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{title} - Top {top_n} Most Frequent Words")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Rank':<6} {'Word':<25} {'Frequency':<15} {'Bar':<30}")
    logger.info("-" * 80)
    
    max_freq = sorted_words[0][1] if sorted_words else 1
    bar_length = 25
    
    for rank, (word, freq) in enumerate(sorted_words, 1):
        bar = "█" * int((freq / max_freq) * bar_length)
        logger.info(f"{rank:<6} {word:<25} {freq:<15,} {bar}")


def display_token_distribution_terminal(token_lengths: List[int], title: str = "Token Distribution") -> None:
    """
    Display token distribution summary in terminal.
    
    Args:
        token_lengths: List of token counts per document
        title: Display title
    """
    if not token_lengths:
        return
    
    import numpy as np
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{title} - Sequence Length Statistics")
    logger.info(f"{'=' * 80}")
    
    logger.info(f"Total Documents: {len(token_lengths):,}")
    logger.info(f"Total Tokens: {sum(token_lengths):,}")
    logger.info(f"Average Tokens per Document: {np.mean(token_lengths):.1f}")
    logger.info(f"Median Tokens: {np.median(token_lengths):.1f}")
    logger.info(f"Min Tokens: {min(token_lengths)}")
    logger.info(f"Max Tokens: {max(token_lengths)}")
    logger.info(f"Std Deviation: {np.std(token_lengths):.1f}")
    logger.info("")
    logger.info("Context Window Thresholds:")
    logger.info(f"  Documents > 1024 tokens: {sum(1 for l in token_lengths if l > 1024):,} ({sum(1 for l in token_lengths if l > 1024)/len(token_lengths)*100:.1f}%)")
    logger.info(f"  Documents > 2048 tokens: {sum(1 for l in token_lengths if l > 2048):,} ({sum(1 for l in token_lengths if l > 2048)/len(token_lengths)*100:.1f}%)")
    logger.info(f"  Documents > 4096 tokens (Llama 2 limit): {sum(1 for l in token_lengths if l > 4096):,} ({sum(1 for l in token_lengths if l > 4096)/len(token_lengths)*100:.1f}%)")


def display_language_distribution_terminal(lang_dist: Dict[str, int], title: str = "Language Distribution") -> None:
    """
    Display language distribution summary in terminal.
    
    Args:
        lang_dist: Dictionary mapping language codes to counts
        title: Display title
    """
    if not lang_dist:
        return
    
    sorted_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
    total = sum(lang_dist.values())
    swahili_count = lang_dist.get('sw', 0)
    swahili_pct = (swahili_count / total * 100) if total > 0 else 0.0
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{title}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total Detected: {total:,}")
    logger.info(f"Swahili (sw): {swahili_count:,} ({swahili_pct:.1f}%)")
    logger.info("")
    logger.info(f"{'Language':<15} {'Count':<15} {'Percentage':<15} {'Bar':<30}")
    logger.info("-" * 80)
    
    max_count = sorted_langs[0][1] if sorted_langs else 1
    bar_length = 25
    
    for lang, count in sorted_langs[:15]:  # Top 15 languages
        pct = (count / total * 100) if total > 0 else 0.0
        bar = "█" * int((count / max_count) * bar_length)
        lang_display = "Swahili" if lang == 'sw' else lang
        logger.info(f"{lang_display:<15} {count:<15,} {pct:<14.1f}% {bar}")


def display_topics_terminal(topics: List[Dict], num_topics: int, title: str = "Topic Modeling") -> None:
    """
    Display topic modeling summary in terminal.
    
    Args:
        topics: List of topic dictionaries with 'topic_id' and 'words'
        num_topics: Number of topics to display
        title: Display title
    """
    if not topics:
        return
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{title} - Top Words per Topic")
    logger.info(f"{'=' * 80}")
    
    for topic in topics[:num_topics]:
        topic_id = topic.get('topic_id', 0)
        words_data = topic.get('words', [])
        
        if not words_data:
            continue
        
        logger.info(f"\nTopic {topic_id}:")
        logger.info("-" * 80)
        logger.info(f"{'Rank':<6} {'Word':<25} {'Weight':<15}")
        logger.info("-" * 80)
        
        for rank, word_data in enumerate(words_data[:10], 1):  # Top 10 words per topic
            word = word_data.get('word', '')
            weight = word_data.get('weight', 0.0)
            logger.info(f"{rank:<6} {word:<25} {weight:<15.4f}")


def display_eda_summary_terminal(eda_results: Dict) -> None:
    """
    Display comprehensive EDA summary in terminal.
    
    Args:
        eda_results: Dictionary containing all EDA analysis results
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPLORATORY DATA ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    # Process each file's results
    for key, results in eda_results.items():
        if key == 'aggregate':
            prefix = 'Aggregate (All Files)'
        else:
            prefix = results.get('filename', key).replace('.txt', '').capitalize()
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"DATASET: {prefix}")
        logger.info(f"{'=' * 80}")
        
        # Vocabulary summary
        vocab = results.get('vocabulary', {})
        if vocab:
            logger.info(f"\nVocabulary Statistics:")
            logger.info(f"  Total Words: {vocab.get('total_words', 0):,}")
            logger.info(f"  Unique Words: {vocab.get('unique_words', 0):,}")
            logger.info(f"  Avg Words per Document: {vocab.get('average_words_per_document', 0):.1f}")
            
            word_freq = vocab.get('word_frequencies', {})
            if word_freq:
                display_word_frequency_terminal(word_freq, top_n=20, title=f"{prefix} - Word Frequency")
        
        # Token summary
        tokens = results.get('tokens', {})
        if tokens and tokens.get('tokenizer_available'):
            token_lengths = tokens.get('token_lengths', [])
            if token_lengths:
                display_token_distribution_terminal(token_lengths, title=f"{prefix} - Token Distribution")
        
        # Language summary
        language = results.get('language', {})
        if language:
            lang_dist = language.get('languages_detected', {})
            if lang_dist:
                display_language_distribution_terminal(lang_dist, title=f"{prefix} - Language Distribution")
        
        # Topics summary
        topics = results.get('topics', {})
        if topics and topics.get('model_available'):
            topic_list = topics.get('topics', [])
            if topic_list:
                display_topics_terminal(topic_list, num_topics=5, title=f"{prefix} - Topics")
    
    logger.info("\n" + "=" * 80)
    logger.info("END OF EDA SUMMARY")
    logger.info("=" * 80)


def generate_all_visualizations(eda_results: Dict, output_dir: str) -> None:
    """
    Generate all visualizations from EDA results.
    
    Args:
        eda_results: Dictionary containing EDA analysis results
        output_dir: Directory to save all visualizations
    """
    logger.info("=" * 80)
    logger.info("Generating all visualizations...")
    logger.info("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each file's results
    for key, results in eda_results.items():
        if key == 'aggregate':
            prefix = 'aggregate'
            title_prefix = 'Aggregate'
        else:
            prefix = results.get('filename', key).replace('.txt', '')
            title_prefix = prefix.capitalize()
        
        logger.info(f"\nGenerating visualizations for {prefix}...")
        
        # Word frequency plot
        vocab = results.get('vocabulary', {})
        word_freq = vocab.get('word_frequencies', {})
        if word_freq:
            freq_path = output_path / f"{prefix}_word_frequency.png"
            plot_word_frequency(word_freq, top_n=30, output_path=str(freq_path),
                              title=f"{title_prefix} Dataset")
        
        # Token distribution plot
        tokens = results.get('tokens', {})
        token_lengths = tokens.get('token_lengths', [])
        if token_lengths:
            token_path = output_path / f"{prefix}_token_distribution.png"
            plot_token_distribution(token_lengths, output_path=str(token_path),
                                  title=f"{title_prefix} Dataset")
        
        # Language distribution plot
        language = results.get('language', {})
        lang_dist = language.get('languages_detected', {})
        if lang_dist:
            lang_path = output_path / f"{prefix}_language_distribution.png"
            plot_language_distribution(lang_dist, output_path=str(lang_path),
                                     title=f"{title_prefix} Dataset")
        
        # Topic visualization
        topics = results.get('topics', {})
        topic_list = topics.get('topics', [])
        if topic_list:
            topic_path = output_path / f"{prefix}_topics.png"
            num_topics = min(9, len(topic_list))  # Display up to 9 topics in grid
            plot_topics(topic_list, num_topics=num_topics, output_path=str(topic_path),
                       title=f"{title_prefix} Dataset")
    
    logger.info("\n" + "=" * 80)
    logger.info("All visualizations generated successfully!")
    logger.info(f"Visualizations saved to: {output_path}")
    logger.info("=" * 80)
    
    # Display comprehensive summary in terminal
    display_eda_summary_terminal(eda_results)

