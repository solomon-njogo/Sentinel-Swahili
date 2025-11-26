"""
Evaluation Utilities for Transformer Models
Model-agnostic evaluation metrics and reporting functions.
"""

import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_accuracy(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score (0-1)
    """
    return accuracy_score(y_true, y_pred)


def calculate_f1_score(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = 'weighted'
) -> float:
    """
    Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('micro', 'macro', 'weighted', 'binary', None)
    
    Returns:
        F1 score
    """
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return f1


def calculate_precision(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = 'weighted'
) -> float:
    """
    Calculate precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('micro', 'macro', 'weighted', 'binary', None)
    
    Returns:
        Precision score
    """
    precision, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return precision


def calculate_recall(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = 'weighted'
) -> float:
    """
    Calculate recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('micro', 'macro', 'weighted', 'binary', None)
    
    Returns:
        Recall score
    """
    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return recall


def calculate_all_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate all metrics (accuracy, precision, recall, F1).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for precision, recall, F1
    
    Returns:
        Dictionary with all metrics
    """
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred, average=average)
    recall = calculate_recall(y_true, y_pred, average=average)
    f1 = calculate_f1_score(y_true, y_pred, average=average)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def get_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    target_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> Union[str, Dict]:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of class names (optional)
        output_dict: If True, return as dictionary; if False, return as string
    
    Returns:
        Classification report (string or dictionary)
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )


def get_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None
) -> np.ndarray:
    """
    Generate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to include (optional)
    
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def print_evaluation_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    label_mapping: Optional[Dict[int, str]] = None,
    dataset_name: str = "Dataset"
):
    """
    Print a comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_mapping: Dictionary mapping integer labels to string labels
        dataset_name: Name of the dataset for display
    """
    print("=" * 80)
    print(f"Evaluation Report: {dataset_name}")
    print("=" * 80)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print("\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    # Classification report
    target_names = None
    if label_mapping:
        # Get sorted list of class names
        sorted_labels = sorted(label_mapping.keys())
        target_names = [label_mapping[label] for label in sorted_labels]
    
    print("\n" + "=" * 80)
    print("Per-Class Metrics:")
    print("=" * 80)
    report = get_classification_report(y_true, y_pred, target_names=target_names, output_dict=False)
    print(report)
    
    # Confusion matrix
    print("\n" + "=" * 80)
    print("Confusion Matrix:")
    print("=" * 80)
    cm = get_confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("=" * 80)


def convert_predictions(
    predictions: Union[List, np.ndarray, Any],
    is_logits: bool = True
) -> np.ndarray:
    """
    Convert model predictions to class labels.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        is_logits: If True, predictions are logits; if False, they are probabilities
    
    Returns:
        Array of predicted class labels
    """
    # Convert to numpy if needed
    if not isinstance(predictions, np.ndarray):
        # Try to convert from tensor
        try:
            import torch
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            else:
                predictions = np.array(predictions)
        except:
            predictions = np.array(predictions)
    
    # Handle logits (apply argmax)
    if is_logits:
        if len(predictions.shape) > 1:
            # Multi-dimensional: take argmax along last dimension
            predictions = np.argmax(predictions, axis=-1)
        else:
            # Already 1D, assume they're already class labels
            pass
    else:
        # Probabilities: take argmax
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=-1)
    
    return predictions


if __name__ == "__main__":
    # Test evaluation utilities
    print("=" * 80)
    print("Testing Evaluation Utilities")
    print("=" * 80)
    
    # Create sample predictions and labels
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 1, 2, 0]
    
    # Calculate individual metrics
    print("\nIndividual Metrics:")
    print(f"Accuracy: {calculate_accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {calculate_precision(y_true, y_pred):.4f}")
    print(f"Recall: {calculate_recall(y_true, y_pred):.4f}")
    print(f"F1-Score: {calculate_f1_score(y_true, y_pred):.4f}")
    
    # Calculate all metrics
    print("\nAll Metrics:")
    metrics = calculate_all_metrics(y_true, y_pred)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Classification report
    print("\n" + "=" * 80)
    print("Classification Report:")
    print("=" * 80)
    report = get_classification_report(y_true, y_pred, output_dict=False)
    print(report)
    
    # Confusion matrix
    print("\n" + "=" * 80)
    print("Confusion Matrix:")
    print("=" * 80)
    cm = get_confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Full evaluation report
    label_mapping = {0: 'class_0', 1: 'class_1', 2: 'class_2'}
    print_evaluation_report(y_true, y_pred, label_mapping, "Test Dataset")
    
    # Test prediction conversion
    print("\n" + "=" * 80)
    print("Testing Prediction Conversion")
    print("=" * 80)
    
    # Simulate logits (3 classes, 5 samples)
    logits = np.array([
        [2.0, 0.5, 0.1],  # Class 0
        [0.3, 2.5, 0.2],  # Class 1
        [0.1, 0.2, 2.8],  # Class 2
        [1.8, 0.4, 0.3],  # Class 0
        [0.2, 1.9, 0.5],  # Class 1
    ])
    
    predictions = convert_predictions(logits, is_logits=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions: {predictions}")
    print(f"Expected: [0, 1, 2, 0, 1]")
    
    print("\n" + "=" * 80)
    print("Evaluation Utilities Test Complete!")
    print("=" * 80)

