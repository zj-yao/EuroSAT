from __future__ import annotations

import argparse
import csv
import itertools
from argparse import Namespace
from pathlib import Path

import numpy as np

from .train import run_experiment
from .utils import ensure_dir, save_json, timestamp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random search for EuroSAT MLP hyperparameters")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--skip_retrain_best", action="store_true")
    return parser


def sample_hyperparameters(trials: int, seed: int) -> list[dict[str, object]]:
    candidates = list(
        itertools.product(
            [0.1, 0.03, 0.01, 0.003],
            [(128, 64), (256, 128)],
            [0.0, 1e-4, 1e-3],
            ["relu", "tanh"],
        )
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    sampled = candidates[: min(trials, len(candidates))]

    configs: list[dict[str, object]] = []
    for lr, hidden_dims, weight_decay, activation in sampled:
        configs.append(
            {
                "lr": lr,
                "hidden_dims": list(hidden_dims),
                "weight_decay": weight_decay,
                "activation": activation,
            }
        )
    return configs


def write_results_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["trial", "lr", "hidden_dims", "weight_decay", "activation", "best_val_accuracy", "best_epoch", "checkpoint"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    run_name = args.run_name or f"search_{timestamp()}"
    search_dir = ensure_dir(Path(args.output_dir) / run_name)

    sampled_configs = sample_hyperparameters(args.trials, args.seed)
    results: list[dict[str, object]] = []

    for index, config in enumerate(sampled_configs, start=1):
        trial_name = f"{run_name}_trial_{index:02d}"
        trial_args = Namespace(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            run_name=trial_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            step_size=args.step_size,
            gamma=args.gamma,
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            image_size=args.image_size,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
        )
        result = run_experiment(trial_args)
        results.append(
            {
                "trial": trial_name,
                "lr": config["lr"],
                "hidden_dims": config["hidden_dims"],
                "weight_decay": config["weight_decay"],
                "activation": config["activation"],
                "best_val_accuracy": result["best_val_accuracy"],
                "best_epoch": result["best_epoch"],
                "checkpoint": result["checkpoint"],
            }
        )

    best_result = max(results, key=lambda item: item["best_val_accuracy"])
    write_results_csv(search_dir / "search_results.csv", results)
    save_json(search_dir / "search_results.json", {"results": results, "best": best_result})

    print(f"Best trial: {best_result['trial']}")
    print(f"Best val accuracy: {best_result['best_val_accuracy']:.4f}")

    if not args.skip_retrain_best:
        retrain_name = f"{run_name}_best"
        retrain_args = Namespace(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            run_name=retrain_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=best_result["lr"],
            weight_decay=best_result["weight_decay"],
            step_size=args.step_size,
            gamma=args.gamma,
            hidden_dims=best_result["hidden_dims"],
            activation=best_result["activation"],
            image_size=args.image_size,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
        )
        retrain_result = run_experiment(retrain_args)
        save_json(
            search_dir / "best_retrain.json",
            {"best_search_trial": best_result, "retrain_result": retrain_result},
        )
        print(f"Retrained best configuration. Checkpoint: {retrain_result['checkpoint']}")


if __name__ == "__main__":
    main()

