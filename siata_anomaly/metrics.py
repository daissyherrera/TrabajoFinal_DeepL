"""Evaluation metrics and visualization for anomaly detection."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score


def precision_recall_f1(y_true, y_pred, probs=None):
    """Compute precision, recall, F1 and optionally AUC-PR.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Binary predictions (0/1).
        probs: Raw probabilities for AUC-PR. Optional.

    Returns:
        Dict with precision, recall, f1, and auc_pr (if probs provided).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    result = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
    if probs is not None:
        if y_true.sum() == 0:
            result['auc_pr'] = float('nan')
        else:
            result['auc_pr'] = float(average_precision_score(y_true, probs))
    return result


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot a 2x2 confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions.
        title: Plot title.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title='Training History'):
    """Plot loss and accuracy curves from a Keras History object.

    Args:
        history: Keras History object returned by model.fit().
        title: Plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='train')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title(f'{title} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    metric = 'accuracy' if 'accuracy' in history.history else 'binary_accuracy'
    if metric in history.history:
        axes[1].plot(history.history[metric], label='train')
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            axes[1].plot(history.history[val_metric], label='val')
        axes[1].set_title(f'{title} — Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

    plt.tight_layout()
    plt.show()


def summary_table(results_dict):
    """Build a DataFrame comparing metrics across models.

    Args:
        results_dict: {model_name: metrics_dict} from evaluate().

    Returns:
        Formatted pandas DataFrame.
    """
    rows = []
    for name, metrics in results_dict.items():
        rows.append({'Model': name, **{k: round(v, 4) for k, v in metrics.items()}})
    return pd.DataFrame(rows).set_index('Model')
