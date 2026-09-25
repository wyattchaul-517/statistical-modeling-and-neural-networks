"""Detailed evaluation and error analysis for MNIST CNN."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect all predictions, true labels, and per-sample confidence.

    Returns
    -------
    images : np.ndarray of shape (N, 1, 28, 28)
    labels : np.ndarray of shape (N,)
    preds  : np.ndarray of shape (N,)
    """
    model.eval()
    all_images, all_labels, all_preds = [], [], []

    for images, labels in loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        _, predicted = outputs.max(1)

        all_images.append(images.cpu().numpy())
        all_labels.append(labels.numpy())
        all_preds.append(predicted.cpu().numpy())

    return (
        np.concatenate(all_images),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )


def confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    """Compute confusion matrix without sklearn dependency."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def per_class_accuracy(
    cm: np.ndarray,
) -> np.ndarray:
    """Per-class accuracy from confusion matrix."""
    return cm.diagonal() / cm.sum(axis=1)