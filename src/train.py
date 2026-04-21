from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .autograd import Tensor
from .dataset import DatasetSplit, batch_iterator, build_splits, compute_channel_stats
from .layers import cross_entropy_from_logits, softmax_cross_entropy
from .metrics import accuracy_score
from .model import MLP
from .optim import SGD, StepLRScheduler
from .utils import ensure_dir, plot_first_layer_weights, plot_training_curves, save_history_csv, save_json, set_seed, timestamp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a three-layer MLP on EuroSAT")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--hidden_dims", type=int, nargs=2, default=[128, 64])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--min_crop_scale", type=float, default=0.85)
    parser.add_argument("--brightness_jitter", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser


def evaluate_split(
    model: MLP,
    split: DatasetSplit,
    data_dir: str | Path,
    batch_size: int,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    losses: list[float] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for features, labels in batch_iterator(
        split,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        mean=mean,
        std=std,
        shuffle=False,
    ):
        logits = model.forward_numpy(features)
        losses.append(cross_entropy_from_logits(logits, labels))
        predictions.append(logits.argmax(axis=1))
        targets.append(labels)

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_checkpoint(
    checkpoint_path: str | Path,
    model: MLP,
    metadata: dict[str, Any],
) -> None:
    payload = model.state_dict()
    payload["metadata"] = np.array(json.dumps(metadata), dtype=object)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, **payload)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)

    run_name = args.run_name or f"train_{timestamp()}"
    run_dir = ensure_dir(Path(args.output_dir) / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    plot_dir = ensure_dir(run_dir / "plots")

    splits, class_names = build_splits(
        args.data_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    mean, std = compute_channel_stats(args.data_dir, train_split.paths, image_size=args.image_size)
    input_dim = args.image_size * args.image_size * 3
    model = MLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=len(class_names),
        activation=args.activation,
    )
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = StepLRScheduler(optimizer, step_size=args.step_size, gamma=args.gamma)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_accuracy = -np.inf
    best_epoch = -1
    best_state_dict: dict[str, np.ndarray] | None = None
    best_checkpoint_path = checkpoint_dir / "best_model.npz"

    for epoch in range(1, args.epochs + 1):
        batch_losses: list[float] = []
        train_targets: list[np.ndarray] = []
        train_predictions: list[np.ndarray] = []

        for features, labels in batch_iterator(
            train_split,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            mean=mean,
            std=std,
            shuffle=True,
            seed=args.seed + epoch,
            augment=args.augment,
            crop_scale_range=(args.min_crop_scale, 1.0),
            brightness_jitter=args.brightness_jitter,
        ):
            inputs = Tensor(features, requires_grad=False)
            logits = model(inputs)
            loss = softmax_cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.data))
            train_targets.append(labels)
            train_predictions.append(logits.data.argmax(axis=1))

        train_y_true = np.concatenate(train_targets)
        train_y_pred = np.concatenate(train_predictions)
        val_metrics = evaluate_split(
            model,
            val_split,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            mean=mean,
            std=std,
        )

        train_loss = float(np.mean(batch_losses))
        train_accuracy = accuracy_score(train_y_true, train_y_pred)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(float(optimizer.lr))

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = model.state_dict()
            metadata = {
                "class_names": class_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "activation": args.activation,
                "hidden_dims": list(args.hidden_dims),
                "input_dim": input_dim,
                "image_size": args.image_size,
                "momentum": args.momentum,
                "augment": args.augment,
                "min_crop_scale": args.min_crop_scale,
                "brightness_jitter": args.brightness_jitter,
                "train_paths": train_split.paths,
                "train_labels": train_split.labels.tolist(),
                "val_paths": val_split.paths,
                "val_labels": val_split.labels.tolist(),
                "test_paths": test_split.paths,
                "test_labels": test_split.labels.tolist(),
                "seed": args.seed,
            }
            save_checkpoint(best_checkpoint_path, model, metadata)

        scheduler.step(epoch)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"lr={optimizer.lr:.5f}"
        )

    plot_training_curves(history, plot_dir / "training_curves.png")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    plot_first_layer_weights(model.fc1.weight.data, args.image_size, plot_dir / "first_layer_weights.png")
    save_json(
        run_dir / "history.json",
        {
            "history": history,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "checkpoint": str(best_checkpoint_path),
        },
    )
    save_history_csv(run_dir / "history.csv", history)

    result = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "checkpoint": str(best_checkpoint_path),
        "best_val_accuracy": float(best_val_accuracy),
        "best_epoch": int(best_epoch),
        "history": history,
    }
    return result


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_experiment(args)
    print(
        f"Finished training. Best val accuracy={result['best_val_accuracy']:.4f} "
        f"at epoch {result['best_epoch']}."
    )
    print(f"Best checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
