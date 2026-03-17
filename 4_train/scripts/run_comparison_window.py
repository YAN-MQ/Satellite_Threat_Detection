#!/usr/bin/env python3
"""Run baseline comparison experiments on the current 3-class window dataset."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    BaselineTrainer,
    DSC_CBAM_GRU,
    DeepLearningBaseline,
    count_flops,
    count_parameters,
)
from src.models.baseline import CNN_LSTM, MLP
from src.models.dsc_cbam_lstm import DSC_CBAM_LSTM

from experiment_utils import (
    add_composite_scores,
    flatten_metrics,
    load_window_data,
    make_window_loaders,
    resolve_device,
    set_seed,
    train_deep_model,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model comparison on dataset_window")
    parser.add_argument("--data_dir", default="../dataset_window")
    parser.add_argument("--output_dir", default="experiments_window/comparison")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--full_hidden_dim", type=int, default=64)
    parser.add_argument("--full_bidirectional", action="store_true", default=True)
    parser.add_argument("--full_dropout", type=float, default=0.2)
    parser.add_argument("--full_lr", type=float, default=1e-3)
    parser.add_argument("--full_weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    (x_train, y_train, x_val, y_val, x_test, y_test), (train_loader, val_loader, test_loader) = make_window_loaders(
        args.data_dir,
        args.batch_size,
    )

    model_builders = [
        (
            "dsc_cbam_gru",
            "deep",
            lambda: DSC_CBAM_GRU(
                args.input_dim,
                args.num_classes,
                hidden_dim=args.full_hidden_dim,
                bidirectional=args.full_bidirectional,
                dropout=args.full_dropout,
            ),
        ),
        ("dsc_cbam_lstm", "deep", lambda: DSC_CBAM_LSTM(args.input_dim, args.num_classes)),
        ("mlp", "deep", lambda: MLP(args.input_dim, args.num_classes)),
        ("cnn_lstm", "deep", lambda: CNN_LSTM(args.input_dim, args.num_classes)),
    ]
    sklearn_builders = [
        ("rf", "traditional", lambda: BaselineTrainer("rf")),
        ("id3", "traditional", lambda: BaselineTrainer("id3")),
    ]

    summary_rows = []
    raw_results = {}

    print("=" * 60)
    print("Window Comparison Study")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    for model_name, family, builder in model_builders:
        print(f"\n[{model_name}]")
        model = builder()
        checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}.pt")
        result = train_deep_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.full_lr if model_name == "dsc_cbam_gru" else args.lr,
            weight_decay=args.full_weight_decay if model_name == "dsc_cbam_gru" else args.weight_decay,
            save_path=checkpoint_path,
        )
        params = count_parameters(model)
        flops = count_flops(model, input_size=(1, 10, args.input_dim))
        summary_rows.append(flatten_metrics(model_name, family, params, flops, result))
        raw_results[model_name] = {
            "metrics": {
                **result["metrics"],
                "confusion_matrix": result["metrics"]["confusion_matrix"].tolist(),
            },
            "history": result["history"],
            "training_time_sec": result["training_time_sec"],
            "inference_time_sec": result["inference_time_sec"],
        }
        print(
            f"  params={params:,} flops={flops:,} "
            f"acc={result['metrics']['accuracy']:.4f} f1={result['metrics']['f1']:.4f}"
        )

    for model_name, family, builder in sklearn_builders:
        print(f"\n[{model_name}]")
        trainer = builder()
        trainer.create_model()
        train_time = trainer.train(x_train, y_train)
        metrics = trainer.evaluate(x_test, y_test)
        result = {
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "confusion_matrix": [],
            },
            "training_time_sec": train_time,
            "inference_time_sec": float(metrics["inference_time"]),
        }
        summary_rows.append(flatten_metrics(model_name, family, "n/a", "n/a", result))
        raw_results[model_name] = result
        print(f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

    summary_rows = add_composite_scores(summary_rows)
    summary_path = os.path.join(args.output_dir, "comparison_summary.csv")
    raw_path = os.path.join(args.output_dir, "comparison_results.json")
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
    write_json(raw_path, raw_results)
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved detailed results to {raw_path}")


if __name__ == "__main__":
    main()
