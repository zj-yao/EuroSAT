from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_history_csv(path: str | Path, history: dict[str, list[float]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(conf_mat: np.ndarray, class_names: list[str], output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(conf_mat, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    figure.colorbar(image, ax=axis)

    for row in range(conf_mat.shape[0]):
        for col in range(conf_mat.shape[1]):
            axis.text(col, row, str(conf_mat[row, col]), ha="center", va="center", fontsize=7)

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_first_layer_weights(
    weights: np.ndarray,
    image_size: int,
    output_path: str | Path,
    max_filters: int = 16,
) -> None:
    num_filters = min(weights.shape[1], max_filters)
    cols = min(4, num_filters)
    rows = int(np.ceil(num_filters / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for index in range(num_filters):
        patch = weights[:, index].reshape(image_size, image_size, 3)
        patch_min = patch.min()
        patch_max = patch.max()
        normalized = (patch - patch_min) / (patch_max - patch_min + 1e-8)
        axes[index].imshow(normalized)
        axes[index].set_title(f"Neuron {index}")
        axes[index].axis("off")

    for axis in axes[num_filters:]:
        axis.axis("off")

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_misclassified_examples(
    images: list[np.ndarray],
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: str | Path,
    max_examples: int = 9,
) -> None:
    num_examples = min(len(images), max_examples)
    if num_examples == 0:
        return

    cols = min(3, num_examples)
    rows = int(np.ceil(num_examples / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for index in range(num_examples):
        axes[index].imshow(np.clip(images[index], 0.0, 1.0))
        axes[index].set_title(
            f"True: {class_names[y_true[index]]}\nPred: {class_names[y_pred[index]]}",
            fontsize=9,
        )
        axes[index].axis("off")

    for axis in axes[num_examples:]:
        axis.axis("off")

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
