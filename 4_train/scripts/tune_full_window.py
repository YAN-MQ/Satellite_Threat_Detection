#!/usr/bin/env python3
"""Hyperparameter search for the DSC-CBAM-GRU full model."""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DSC_CBAM_GRU, count_flops, count_parameters

from experiment_utils import (
    flatten_metrics,
    make_window_loaders,
    resolve_device,
    set_seed,
    train_deep_model,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune DSC-CBAM-GRU on dataset_window")
    parser.add_argument("--data_dir", default="../dataset_window")
    parser.add_argument("--output_dir", default="experiments_window/full_tuning")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_runs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    _, (train_loader, val_loader, test_loader) = make_window_loaders(args.data_dir, args.batch_size)

    search_space = list(
        itertools.product(
            [64, 96],
            [True],
            [0.2, 0.3],
            [1e-3, 5e-4],
            [1e-2, 5e-3],
        )
    )
    if args.max_runs is not None:
        search_space = search_space[: args.max_runs]

    rows = []
    raw = {}
    best_name = None
    best_acc = -1.0

    print("=" * 60)
    print("Full Model Hyperparameter Search")
    print("=" * 60)

    for idx, (hidden_dim, bidirectional, dropout, lr, weight_decay) in enumerate(search_space, start=1):
        name = f"full_h{hidden_dim}_{'bi' if bidirectional else 'uni'}_d{str(dropout).replace('.', '')}_lr{str(lr).replace('.', '')}_wd{str(weight_decay).replace('.', '')}"
        print(f"\n[{idx}/{len(search_space)}] {name}")
        model = DSC_CBAM_GRU(
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        result = train_deep_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=lr,
            weight_decay=weight_decay,
            save_path=os.path.join(checkpoints_dir, f"{name}.pt"),
        )
        params = count_parameters(model)
        flops = count_flops(model, input_size=(1, 10, args.input_dim))
        row = flatten_metrics(name, "full_tuning", params, flops, result)
        row.update(
            {
                "hidden_dim": hidden_dim,
                "bidirectional": bidirectional,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "val_best": max(result["history"]["val_acc"]) if result["history"]["val_acc"] else None,
            }
        )
        rows.append(row)
        raw[name] = {
            "config": {
                "hidden_dim": hidden_dim,
                "bidirectional": bidirectional,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            "history": result["history"],
            "metrics": {
                **result["metrics"],
                "confusion_matrix": result["metrics"]["confusion_matrix"].tolist(),
            },
            "training_time_sec": result["training_time_sec"],
            "inference_time_sec": result["inference_time_sec"],
        }
        test_acc = result["metrics"]["accuracy"]
        print(f"  acc={test_acc:.4f} f1={result['metrics']['f1']:.4f} params={params:,}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_name = name

        rows.sort(key=lambda item: item["accuracy"], reverse=True)
        summary_path = os.path.join(args.output_dir, "full_tuning_summary.csv")
        detail_path = os.path.join(args.output_dir, "full_tuning_results.json")
        write_csv(
            summary_path,
            rows,
            [
                "model",
                "family",
                "hidden_dim",
                "bidirectional",
                "dropout",
                "lr",
                "weight_decay",
                "params",
                "flops",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "val_best",
                "training_time_sec",
                "inference_time_sec",
                "confusion_matrix",
            ],
        )
        write_json(
            detail_path,
            {
                "best_model": best_name,
                "best_accuracy": best_acc,
                "runs": raw,
            },
        )
    summary_path = os.path.join(args.output_dir, "full_tuning_summary.csv")
    detail_path = os.path.join(args.output_dir, "full_tuning_results.json")
    print(f"\nBest config: {best_name} acc={best_acc:.4f}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved details to {detail_path}")


if __name__ == "__main__":
    main()
