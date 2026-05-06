#!/usr/bin/env python3
"""Run ablation experiments on the selected dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import AblationFactory, count_flops, count_parameters

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


ABLATION_MODELS = [
    ("dsc_cbam_gru", "DSC-CBAM-GRU"),
    ("ablation_no_dsc", "No DSC"),
    ("ablation_no_cbam", "No CBAM"),
    ("ablation_no_gru", "No GRU"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study on the selected dataset")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--output_dir", default="experiments/ablation_formal_tuned")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--full_hidden_dim", type=int, default=None)
    parser.add_argument("--full_bidirectional", action="store_true", default=False)
    parser.add_argument("--full_dropout", type=float, default=None)
    parser.add_argument("--full_lr", type=float, default=None)
    parser.add_argument("--full_weight_decay", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparison_config",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "experiments",
            "comparison_formal_tuned",
            "comparison_results.json",
        ),
    )
    return parser.parse_args()


def load_comparison_payload(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_comparison_full_config(config_path: str) -> dict[str, float | int | bool]:
    return load_comparison_payload(config_path)["_full_model_config"]


def main() -> None:
    args = parse_args()
    comparison_payload = load_comparison_payload(args.comparison_config)
    comparison_full_config = comparison_payload["_full_model_config"]
    comparison_main_result = comparison_payload["dsc_cbam_gru"]
    full_hidden_dim = int(comparison_full_config["hidden_dim"]) if args.full_hidden_dim is None else args.full_hidden_dim
    full_bidirectional = bool(comparison_full_config["bidirectional"]) if not args.full_bidirectional else args.full_bidirectional
    full_dropout = float(comparison_full_config["dropout"]) if args.full_dropout is None else args.full_dropout
    full_conv_dim = int(comparison_full_config["conv_dim"])
    full_dsc_dim = int(comparison_full_config["dsc_dim"])
    full_lr = float(comparison_full_config["lr"]) if args.full_lr is None else args.full_lr
    full_weight_decay = float(comparison_full_config["weight_decay"]) if args.full_weight_decay is None else args.full_weight_decay
    set_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    _, (train_loader, val_loader, test_loader) = make_window_loaders(
        args.data_dir,
        args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    summary_rows = []
    raw_results = {}

    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Comparison config: {args.comparison_config}")

    for model_key, model_label in ABLATION_MODELS:
        print(f"\n[{model_key}] {model_label}")
        if model_key == "dsc_cbam_gru":
            model = AblationFactory.create(
                model_key,
                input_dim=args.input_dim,
                num_classes=args.num_classes,
                hidden_dim=int(comparison_full_config["hidden_dim"]),
                bidirectional=bool(comparison_full_config["bidirectional"]),
                dropout=float(comparison_full_config["dropout"]),
                conv_dim=full_conv_dim,
                dsc_dim=full_dsc_dim,
            )
            lr = float(comparison_full_config["lr"])
            weight_decay = float(comparison_full_config["weight_decay"])
        else:
            model = AblationFactory.create(
                model_key,
                input_dim=args.input_dim,
                num_classes=args.num_classes,
                hidden_dim=full_hidden_dim,
                bidirectional=full_bidirectional,
                dropout=full_dropout,
                conv_dim=full_conv_dim,
                dsc_dim=full_dsc_dim,
            )
            lr = full_lr
            weight_decay = full_weight_decay
        params = count_parameters(model)
        flops = count_flops(model, input_size=(1, 10, args.input_dim))
        checkpoint_path = os.path.join(checkpoints_dir, f"{model_key}.pt")
        if model_key == "dsc_cbam_gru":
            result = {
                "history": comparison_main_result["history"],
                "metrics": {
                    **comparison_main_result["metrics"],
                    "confusion_matrix": np.array(comparison_main_result["metrics"]["confusion_matrix"]),
                },
                "training_time_sec": comparison_main_result["training_time_sec"],
                "inference_time_sec": comparison_main_result["inference_time_sec"],
            }
        else:
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

    summary_rows = add_metric_ranks(summary_rows)
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
            "accuracy_rank",
            "f1_rank",
            "composite_score",
            "composite_rank",
            "confusion_matrix",
        ],
    )
    serializable = {
        "_comparison_config_path": args.comparison_config,
        "_full_model_config_source": "comparison_formal_tuned",
        "_full_model_config": {
            "hidden_dim": int(full_hidden_dim),
            "bidirectional": bool(full_bidirectional),
            "dropout": float(full_dropout),
            "conv_dim": int(full_conv_dim),
            "dsc_dim": int(full_dsc_dim),
            "lr": float(full_lr),
            "weight_decay": float(full_weight_decay),
            "epochs": int(comparison_full_config["epochs"]),
        }
    }
    serializable.update({
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
    })
    write_json(raw_path, serializable)
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved detailed results to {raw_path}")


if __name__ == "__main__":
    main()
