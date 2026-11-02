"""
Evaluation metrics for rugby play classification.

Provides consistent metrics calculation across all models:
- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Confusion matrix
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
PLAY_CATEGORIES = {0: 'scrum', 1: 'line', 2: 'other'}

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str='weighted') -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
    
    Returns:
        Dictionary with accuracy, precision, recall, and f1-score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1)}

def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each play category.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary mapping category names to their metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    for idx, category in PLAY_CATEGORIES.items():
        per_class[category] = {'precision': float(precision[idx]), 'recall': float(recall[idx]), 'f1_score': float(f1[idx]), 'support': int(support[idx])}
    return per_class

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate formatted classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Formatted classification report string
    """
    target_names = [PLAY_CATEGORIES[i] for i in sorted(PLAY_CATEGORIES.keys())]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool=True) -> Dict[str, any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with all metrics and confusion matrix
    """
    results = {'overall': calculate_metrics(y_true, y_pred), 'per_class': calculate_per_class_metrics(y_true, y_pred), 'confusion_matrix': get_confusion_matrix(y_true, y_pred)}
    if verbose:
        print('\n' + '=' * 60)
        print('EVALUATION RESULTS')
        print('=' * 60)
        print(f'\nOverall Metrics:')
        for metric, value in results['overall'].items():
            print(f'  {metric.capitalize()}: {value:.4f}')
        print(f'\nPer-Class Metrics:')
        for category, metrics in results['per_class'].items():
            print(f'\n  {category.upper()}:')
            for metric, value in metrics.items():
                if metric == 'support':
                    print(f'    {metric}: {value}')
                else:
                    print(f'    {metric}: {value:.4f}')
        print(f'\nConfusion Matrix:')
        print(results['confusion_matrix'])
        print('\n' + '=' * 60 + '\n')
    return results
