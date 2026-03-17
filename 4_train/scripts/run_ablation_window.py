#!/usr/bin/env python3
"""Run GRU ablation experiments on the current 3-class window dataset."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import AblationFactory, count_flops, count_parameters

from experiment_utils import (
    add_composite_scores,
    flatten_metrics,
    make_window_loaders,
    resolve_device,
    set_seed,
    train_deep_model,
    write_csv,
    write_json,
)


ABLATION_MODELS = [
    ("full", "DSC-CBAM-GRU"),
    ("ablation_no_dsc", "No DSC"),
    ("ablation_no_cbam", "No CBAM"),
    ("ablation_no_gru", "No GRU"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study on dataset_window")
    parser.add_argument("--data_dir", default="../dataset_window")
    parser.add_argument("--output_dir", default="experiments_window/ablation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--full_hidden_dim", type=int, default=64)
    parser.add_argument("--full_bidirectional", action="store_true", default=True)
    parser.add_argument("--full_dropout", type=float, default=0.3)
    parser.add_argument("--full_lr", type=float, default=1e-4)
    parser.add_argument("--full_weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    _, (train_loader, val_loader, test_loader) = make_window_loaders(args.data_dir, args.batch_size)

    summary_rows = []
    raw_results = {}

    print("=" * 60)
    print("Window Ablation Study")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    for model_key, model_label in ABLATION_MODELS:
        print(f"\n[{model_key}] {model_label}")
        if model_key == "full":
            model = AblationFactory.create(
                model_key,
                input_dim=args.input_dim,
                num_classes=args.num_classes,
                hidden_dim=args.full_hidden_dim,
                bidirectional=args.full_bidirectional,
                dropout=args.full_dropout,
            )
            lr = args.full_lr
            weight_decay = args.full_weight_decay
        else:
            model = AblationFactory.create(model_key, input_dim=args.input_dim, num_classes=args.num_classes)
            lr = args.lr
            weight_decay = args.weight_decay
        params = count_parameters(model)
        flops = count_flops(model, input_size=(1, 10, args.input_dim))
        checkpoint_path = os.path.join(checkpoints_dir, f"{model_key}.pt")
        result = train_deep_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=lr,
            weight_decay=weight_decay,
            save_path=checkpoint_path,
        )
        summary_rows.append(flatten_metrics(model_key, "ablation", params, flops, result))
        raw_results[model_key] = result
        print(
            f"  params={params:,} flops={flops:,} "
            f"acc={result['metrics']['accuracy']:.4f} f1={result['metrics']['f1']:.4f}"
        )

    summary_rows = add_composite_scores(summary_rows)
    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    raw_path = os.path.join(args.output_dir, "ablation_results.json")
    write_csv(
        summary_path,
        summary_rows,
        [
            "model",
            "family",
            "params",
            "flops",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "training_time_sec",
            "inference_time_sec",
            "composite_score",
            "composite_rank",
            "confusion_matrix",
        ],
    )
    serializable = {
        key: {
            "history": value["history"],
            "metrics": {
                **value["metrics"],
                "confusion_matrix": value["metrics"]["confusion_matrix"].tolist(),
            },
            "training_time_sec": value["training_time_sec"],
            "inference_time_sec": value["inference_time_sec"],
        }
        for key, value in raw_results.items()
    }
    write_json(raw_path, serializable)
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved detailed results to {raw_path}")


if __name__ == "__main__":
    main()
