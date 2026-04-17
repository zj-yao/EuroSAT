from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetSplit, batch_iterator
from .layers import cross_entropy_from_logits
from .metrics import accuracy_score, confusion_matrix
from .model import MLP
from .utils import ensure_dir, plot_confusion_matrix, plot_misclassified_examples, save_json, timestamp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained EuroSAT MLP checkpoint")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser


def load_checkpoint(path: str | Path) -> tuple[MLP, dict[str, Any]]:
    checkpoint = np.load(path, allow_pickle=True)
    metadata = json.loads(checkpoint["metadata"].item())
    model = MLP(
        input_dim=int(metadata["input_dim"]),
        hidden_dims=metadata["hidden_dims"],
        num_classes=len(metadata["class_names"]),
        activation=metadata["activation"],
    )
    state_dict = {key: checkpoint[key] for key in checkpoint.files if key != "metadata"}
    model.load_state_dict(state_dict)
    return model, metadata


def evaluate_with_examples(
    model: MLP,
    split: DatasetSplit,
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    mis_images: list[np.ndarray] = []
    mis_true: list[int] = []
    mis_pred: list[int] = []

    for features, labels, images in batch_iterator(
        split,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        mean=mean,
        std=std,
        shuffle=False,
        return_images=True,
    ):
        logits = model.forward_numpy(features)
        preds = logits.argmax(axis=1)
        losses.append(cross_entropy_from_logits(logits, labels))
        predictions.append(preds)
        targets.append(labels)

        mask = preds != labels
        for image, true_label, pred_label in zip(images[mask], labels[mask], preds[mask]):
            if len(mis_images) >= 9:
                break
            mis_images.append(image)
            mis_true.append(int(true_label))
            mis_pred.append(int(pred_label))

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "mis_images": mis_images,
        "mis_true": mis_true,
        "mis_pred": mis_pred,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    run_name = args.run_name or f"test_{timestamp()}"
    run_dir = ensure_dir(Path(args.output_dir) / run_name)
    plot_dir = ensure_dir(run_dir / "plots")

    model, metadata = load_checkpoint(args.checkpoint)
    mean = np.array(metadata["mean"], dtype=np.float32)
    std = np.array(metadata["std"], dtype=np.float32)
    split = DatasetSplit(paths=metadata[f"{args.split}_paths"], labels=np.array(metadata[f"{args.split}_labels"], dtype=np.int64))

    results = evaluate_with_examples(
        model,
        split,
        data_dir=args.data_dir,
        image_size=int(metadata["image_size"]),
        batch_size=args.batch_size,
        mean=mean,
        std=std,
    )
    conf_mat = confusion_matrix(results["y_true"], results["y_pred"], num_classes=len(metadata["class_names"]))
    plot_confusion_matrix(conf_mat, metadata["class_names"], plot_dir / "confusion_matrix.png")
    plot_misclassified_examples(
        results["mis_images"],
        results["mis_true"],
        results["mis_pred"],
        metadata["class_names"],
        plot_dir / "misclassified_examples.png",
    )

    save_json(
        run_dir / "metrics.json",
        {
            "split": args.split,
            "loss": results["loss"],
            "accuracy": results["accuracy"],
            "confusion_matrix": conf_mat.tolist(),
            "checkpoint": args.checkpoint,
        },
    )

    print(f"{args.split} loss={results['loss']:.4f}")
    print(f"{args.split} accuracy={results['accuracy']:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
