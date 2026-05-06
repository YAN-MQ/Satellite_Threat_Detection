#!/usr/bin/env python3
"""Run baseline comparison experiments on the selected dataset."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DSC_CBAM_GRU, count_flops, count_parameters
from src.models.baseline import CNN_LSTM, MLP
from src.models.dsc_cbam_lstm import DSC_CBAM_LSTM

from experiment_utils import (
    add_composite_scores,
    add_metric_ranks,
    flatten_metrics,
    make_window_loaders,
    resolve_device,
    set_seed,
    train_deep_model,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model comparison on the selected dataset")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--output_dir", default="experiments/comparison_formal_tuned")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--full_hidden_dim", type=int, default=32)
    parser.add_argument("--full_bidirectional", action="store_true", default=False)
    parser.add_argument("--full_dropout", type=float, default=0.4)
    parser.add_argument("--full_conv_dim", type=int, default=16)
    parser.add_argument("--full_dsc_dim", type=int, default=48)
    parser.add_argument("--full_lr", type=float, default=3e-4)
    parser.add_argument("--full_weight_decay", type=float, default=1e-2)
    parser.add_argument("--baseline_dropout", type=float, default=0.3)
    parser.add_argument("--baseline_lr", type=float, default=1e-4)
    parser.add_argument("--baseline_weight_decay", type=float, default=1e-2)
    parser.add_argument("--lstm_hidden_dim", type=int, default=72)
    parser.add_argument("--lstm_bidirectional", action="store_true", default=False)
    parser.add_argument("--cnn_hidden_dim", type=int, default=64)
    parser.add_argument("--cnn_conv1", type=int, default=40)
    parser.add_argument("--cnn_conv2", type=int, default=80)
    parser.add_argument("--include_mlp", action="store_true", default=False)
    parser.add_argument("--include_traditional", action="store_true", default=False)
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
        max_samples=args.max_samples,
        seed=args.seed,
    )

    full_model_config = {
        "hidden_dim": args.full_hidden_dim,
        "bidirectional": bool(args.full_bidirectional),
        "dropout": args.full_dropout,
        "conv_dim": args.full_conv_dim,
        "dsc_dim": args.full_dsc_dim,
        "lr": args.full_lr,
        "weight_decay": args.full_weight_decay,
        "epochs": args.epochs,
    }
    baseline_common_config = {
        "dropout": args.baseline_dropout,
        "lr": args.baseline_lr,
        "weight_decay": args.baseline_weight_decay,
        "epochs": args.epochs,
    }

    model_builders = [
        {
            "name": "dsc_cbam_gru",
            "family": "deep",
            "train_config": {
                "lr": full_model_config["lr"],
                "weight_decay": full_model_config["weight_decay"],
            },
            "raw_config_key": "_full_model_config",
            "raw_config": full_model_config,
            "builder": lambda: DSC_CBAM_GRU(
                args.input_dim,
                args.num_classes,
                hidden_dim=full_model_config["hidden_dim"],
                bidirectional=full_model_config["bidirectional"],
                dropout=full_model_config["dropout"],
                conv_dim=full_model_config["conv_dim"],
                dsc_dim=full_model_config["dsc_dim"],
            ),
        },
        {
            "name": "dsc_cbam_lstm",
            "family": "deep",
            "train_config": {
                "lr": baseline_common_config["lr"],
                "weight_decay": baseline_common_config["weight_decay"],
            },
            "raw_config": {
                "hidden_dim": args.lstm_hidden_dim,
                "bidirectional": bool(args.lstm_bidirectional),
                **baseline_common_config,
            },
            "builder": lambda: DSC_CBAM_LSTM(
                args.input_dim,
                args.num_classes,
                hidden_dim=args.lstm_hidden_dim,
                bidirectional=args.lstm_bidirectional,
                dropout=baseline_common_config["dropout"],
            ),
        },
        {
            "name": "cnn_lstm",
            "family": "deep",
            "train_config": {
                "lr": baseline_common_config["lr"],
                "weight_decay": baseline_common_config["weight_decay"],
            },
            "raw_config": {
                "hidden_dim": args.cnn_hidden_dim,
                "conv_channels": [args.cnn_conv1, args.cnn_conv2],
                "bidirectional": False,
                **baseline_common_config,
            },
            "builder": lambda: CNN_LSTM(
                args.input_dim,
                args.num_classes,
                conv_channels=(args.cnn_conv1, args.cnn_conv2),
                hidden_dim=args.cnn_hidden_dim,
                bidirectional=False,
                dropout=baseline_common_config["dropout"],
            ),
        },
    ]
    if args.include_mlp:
        model_builders.append(
            {
                "name": "mlp",
                "family": "deep",
                "train_config": {
                    "lr": baseline_common_config["lr"],
                    "weight_decay": baseline_common_config["weight_decay"],
                },
                "raw_config": {
                    "hidden_dims": [256, 128, 64],
                    **baseline_common_config,
                },
                "builder": lambda: MLP(
                    args.input_dim,
                    args.num_classes,
                    hidden_dims=(256, 128, 64),
                    dropout=baseline_common_config["dropout"],
                ),
            }
        )

    summary_rows = []
    raw_results = {
        "_full_model_config": full_model_config,
        "_baseline_config": {
            "dsc_cbam_lstm": {
                "hidden_dim": args.lstm_hidden_dim,
                "bidirectional": bool(args.lstm_bidirectional),
                **baseline_common_config,
            },
            "cnn_lstm": {
                "hidden_dim": args.cnn_hidden_dim,
                "conv_channels": [args.cnn_conv1, args.cnn_conv2],
                "bidirectional": False,
                **baseline_common_config,
            },
            **(
                {
                    "mlp": {
                        "hidden_dims": [256, 128, 64],
                        **baseline_common_config,
                    }
                }
                if args.include_mlp
                else {}
            ),
        },
    }

    print("=" * 60)
    print("Model Comparison Study")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    for model_spec in model_builders:
        model_name = model_spec["name"]
        family = model_spec["family"]
        print(f"\n[{model_name}]")
        model = model_spec["builder"]()
        checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}.pt")
        result = train_deep_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=model_spec["train_config"]["lr"],
            weight_decay=model_spec["train_config"]["weight_decay"],
            save_path=checkpoint_path,
        )
        params = count_parameters(model)
        flops = count_flops(model, input_size=(1, 10, args.input_dim))
        summary_rows.append(flatten_metrics(model_name, family, params, flops, result))
        raw_results[model_name] = {
            "config": model_spec["raw_config"],
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

    if args.include_traditional:
        from src.models import BaselineTrainer

        sklearn_builders = [
            ("rf", "traditional", lambda: BaselineTrainer("rf")),
            ("id3", "traditional", lambda: BaselineTrainer("id3")),
        ]

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

    summary_rows = add_metric_ranks(summary_rows)
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
            "accuracy_rank",
            "f1_rank",
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
