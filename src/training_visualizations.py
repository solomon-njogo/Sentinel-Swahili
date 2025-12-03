"""
Training Visualization Module
Generates plots for training metrics (loss curves, perplexity, learning rate).
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def extract_training_history(
    training_history: List[Dict[str, Any]]
) -> Dict[str, List[float]]:
    """
    Extract training metrics from training history.
    
    Args:
        training_history: List of log dictionaries from Trainer
        
    Returns:
        Dictionary with lists of metric values
    """
    metrics = {
        "train_loss": [],
        "eval_loss": [],
        "learning_rate": [],
        "epoch": [],
        "step": []
    }
    
    for entry in training_history:
        if "loss" in entry:
            metrics["train_loss"].append((entry.get("step", 0), entry["loss"]))
        
        if "eval_loss" in entry:
            metrics["eval_loss"].append((entry.get("step", 0), entry["eval_loss"]))
        
        if "learning_rate" in entry:
            metrics["learning_rate"].append((entry.get("step", 0), entry["learning_rate"]))
        
        if "epoch" in entry:
            metrics["epoch"].append((entry.get("step", 0), entry["epoch"]))
    
    # Convert to separate lists
    result = {}
    for key, values in metrics.items():
        if values:
            steps, vals = zip(*values)
            result[f"{key}_steps"] = list(steps)
            result[key] = list(vals)
    
    return result


def plot_training_loss(
    training_history: List[Dict[str, Any]],
    output_path: str,
    title: str = "Training Loss"
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        training_history: Training history from Trainer
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info("Generating training loss plot...")
    
    metrics = extract_training_history(training_history)
    
    if not metrics.get("train_loss"):
        logger.warning("No training loss data available")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Plot training loss
    if "train_loss" in metrics:
        plt.plot(
            metrics["train_loss_steps"],
            metrics["train_loss"],
            label="Training Loss",
            color="steelblue",
            linewidth=2,
            alpha=0.8
        )
    
    # Plot validation loss if available
    if "eval_loss" in metrics:
        plt.plot(
            metrics["eval_loss_steps"],
            metrics["eval_loss"],
            label="Validation Loss",
            color="coral",
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=4
        )
    
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training loss plot saved to: {output_path}")


def plot_perplexity(
    training_history: List[Dict[str, Any]],
    output_path: str,
    title: str = "Perplexity"
) -> None:
    """
    Plot perplexity over training steps.
    
    Args:
        training_history: Training history from Trainer
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info("Generating perplexity plot...")
    
    metrics = extract_training_history(training_history)
    
    if not metrics.get("eval_loss"):
        logger.warning("No evaluation loss data available for perplexity")
        return
    
    import numpy as np
    
    # Compute perplexity from eval loss
    perplexity = [np.exp(loss) for loss in metrics["eval_loss"]]
    
    plt.figure(figsize=(14, 8))
    plt.plot(
        metrics["eval_loss_steps"],
        perplexity,
        label="Validation Perplexity",
        color="purple",
        linewidth=2,
        alpha=0.8,
        marker="s",
        markersize=4
    )
    
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale for perplexity
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Perplexity plot saved to: {output_path}")


def plot_learning_rate(
    training_history: List[Dict[str, Any]],
    output_path: str,
    title: str = "Learning Rate Schedule"
) -> None:
    """
    Plot learning rate schedule over training steps.
    
    Args:
        training_history: Training history from Trainer
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info("Generating learning rate plot...")
    
    metrics = extract_training_history(training_history)
    
    if not metrics.get("learning_rate"):
        logger.warning("No learning rate data available")
        return
    
    plt.figure(figsize=(14, 8))
    plt.plot(
        metrics["learning_rate_steps"],
        metrics["learning_rate"],
        label="Learning Rate",
        color="green",
        linewidth=2,
        alpha=0.8
    )
    
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale for learning rate
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Learning rate plot saved to: {output_path}")


def plot_training_summary(
    training_history: List[Dict[str, Any]],
    output_path: str,
    title: str = "Training Summary"
) -> None:
    """
    Create a comprehensive training summary plot with multiple subplots.
    
    Args:
        training_history: Training history from Trainer
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info("Generating training summary plot...")
    
    metrics = extract_training_history(training_history)
    
    if not metrics:
        logger.warning("No training metrics available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    import numpy as np
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    if "train_loss" in metrics:
        ax1.plot(
            metrics["train_loss_steps"],
            metrics["train_loss"],
            label="Training Loss",
            color="steelblue",
            linewidth=2
        )
    if "eval_loss" in metrics:
        ax1.plot(
            metrics["eval_loss_steps"],
            metrics["eval_loss"],
            label="Validation Loss",
            color="coral",
            linewidth=2,
            marker="o",
            markersize=3
        )
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Perplexity
    ax2 = axes[0, 1]
    if "eval_loss" in metrics:
        perplexity = [np.exp(loss) for loss in metrics["eval_loss"]]
        ax2.plot(
            metrics["eval_loss_steps"],
            perplexity,
            label="Validation Perplexity",
            color="purple",
            linewidth=2,
            marker="s",
            markersize=3
        )
        ax2.set_yscale("log")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax3 = axes[1, 0]
    if "learning_rate" in metrics:
        ax3.plot(
            metrics["learning_rate_steps"],
            metrics["learning_rate"],
            label="Learning Rate",
            color="green",
            linewidth=2
        )
        ax3.set_yscale("log")
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Epoch Progress
    ax4 = axes[1, 1]
    if "epoch" in metrics:
        ax4.plot(
            metrics["epoch_steps"],
            metrics["epoch"],
            label="Epoch",
            color="orange",
            linewidth=2
        )
    ax4.set_xlabel("Training Step")
    ax4.set_ylabel("Epoch")
    ax4.set_title("Training Progress")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training summary plot saved to: {output_path}")


def generate_all_training_visualizations(
    training_history: List[Dict[str, Any]],
    output_dir: str = "reports"
) -> None:
    """
    Generate all training visualizations.
    
    Args:
        training_history: Training history from Trainer
        output_dir: Directory to save plots
    """
    logger.info("=" * 80)
    logger.info("Generating Training Visualizations")
    logger.info("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate individual plots
    plot_training_loss(
        training_history,
        str(output_path / "training_loss.png"),
        "Training and Validation Loss"
    )
    
    plot_perplexity(
        training_history,
        str(output_path / "perplexity.png"),
        "Validation Perplexity"
    )
    
    plot_learning_rate(
        training_history,
        str(output_path / "learning_rate.png"),
        "Learning Rate Schedule"
    )
    
    # Generate summary plot
    plot_training_summary(
        training_history,
        str(output_path / "training_summary.png"),
        "Training Summary"
    )
    
    logger.info("All training visualizations generated successfully!")

