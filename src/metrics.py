from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix

