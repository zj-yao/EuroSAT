from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image


@dataclass
class DatasetSplit:
    paths: list[str]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


def discover_dataset(data_dir: str | Path) -> tuple[list[str], np.ndarray, list[str]]:
    root = Path(data_dir)
    class_names = sorted([path.name for path in root.iterdir() if path.is_dir()])
    if not class_names:
        raise FileNotFoundError(f"No class folders found in {root}")

    all_paths: list[str] = []
    all_labels: list[int] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = root / class_name
        image_paths = sorted([path for path in class_dir.iterdir() if path.is_file()])
        for image_path in image_paths:
            all_paths.append(image_path.relative_to(root).as_posix())
            all_labels.append(class_index)
    return all_paths, np.array(all_labels, dtype=np.int64), class_names


def stratified_split(
    paths: Sequence[str],
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, DatasetSplit]:
    rng = np.random.default_rng(seed)
    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []
    train_labels: list[int] = []
    val_labels: list[int] = []
    test_labels: list[int] = []

    for class_index in sorted(np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == class_index)
        rng.shuffle(class_indices)
        num_samples = len(class_indices)
        num_train = int(num_samples * train_ratio)
        num_val = int(num_samples * val_ratio)
        num_test = num_samples - num_train - num_val

        train_idx = class_indices[:num_train]
        val_idx = class_indices[num_train : num_train + num_val]
        test_idx = class_indices[num_train + num_val : num_train + num_val + num_test]

        train_paths.extend(paths[index] for index in train_idx)
        val_paths.extend(paths[index] for index in val_idx)
        test_paths.extend(paths[index] for index in test_idx)
        train_labels.extend(labels[train_idx].tolist())
        val_labels.extend(labels[val_idx].tolist())
        test_labels.extend(labels[test_idx].tolist())

    return {
        "train": DatasetSplit(train_paths, np.array(train_labels, dtype=np.int64)),
        "val": DatasetSplit(val_paths, np.array(val_labels, dtype=np.int64)),
        "test": DatasetSplit(test_paths, np.array(test_labels, dtype=np.int64)),
    }


def limit_split(split: DatasetSplit, max_samples: int | None, seed: int) -> DatasetSplit:
    if max_samples is None or max_samples >= len(split):
        return split
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(split), size=max_samples, replace=False)
    indices.sort()
    return DatasetSplit([split.paths[index] for index in indices], split.labels[indices])


def build_splits(
    data_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[dict[str, DatasetSplit], list[str]]:
    paths, labels, class_names = discover_dataset(data_dir)
    splits = stratified_split(paths, labels, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    splits["train"] = limit_split(splits["train"], max_train_samples, seed)
    splits["val"] = limit_split(splits["val"], max_val_samples, seed + 1)
    splits["test"] = limit_split(splits["test"], max_test_samples, seed + 2)
    return splits, class_names


def load_image(path: str | Path, image_size: int | None = None) -> np.ndarray:
    image_path = Path(path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def compute_channel_stats(
    data_dir: str | Path,
    relative_paths: Sequence[str],
    image_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_dir)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for relative_path in relative_paths:
        image = load_image(root / relative_path, image_size=image_size)
        flat = image.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sq_sum += np.square(flat).sum(axis=0)
        total_pixels += flat.shape[0]

    mean = channel_sum / total_pixels
    var = channel_sq_sum / total_pixels - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def preprocess_images(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (images - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    return normalized.reshape(images.shape[0], -1).astype(np.float32)


def batch_iterator(
    split: DatasetSplit,
    data_dir: str | Path,
    batch_size: int,
    image_size: int | None,
    mean: np.ndarray,
    std: np.ndarray,
    shuffle: bool = False,
    seed: int = 42,
    return_images: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]]:
    root = Path(data_dir)
    indices = np.arange(len(split))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = [load_image(root / split.paths[index], image_size=image_size) for index in batch_indices]
        image_batch = np.stack(images).astype(np.float32)
        features = preprocess_images(image_batch, mean=mean, std=std)
        label_batch = split.labels[batch_indices]
        if return_images:
            yield features, label_batch, image_batch
        else:
            yield features, label_batch

