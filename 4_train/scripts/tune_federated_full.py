#!/usr/bin/env python3
"""Grid search the OrbitShield_FL federated method."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from federated import FederatedConfig, run_federated_training
from federated.partition import load_window_dataset
from experiment_utils import write_csv, write_json


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated float list."""
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated int list."""
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def format_run_name(beta: float, warmup_rounds: int, global_momentum: float) -> str:
    """Create a stable directory-friendly run name."""
    beta_tag = str(beta).replace(".", "")
    momentum_tag = str(global_momentum).replace(".", "")
    return f"OrbitShield_FL_beta{beta_tag}_warm{warmup_rounds}_gm{momentum_tag}"


def summarize_result(
    run_name: str,
    beta: float,
    warmup_rounds: int,
    global_momentum: float,
    result: dict[str, object],
) -> dict[str, object]:
    """Build one tuning summary row."""
    metrics = result["final_test_metrics"]
    return {
        "variant_name": run_name,
        "method_name": "OrbitShield_FL",
        "beta": beta,
        "warmup_rounds": warmup_rounds,
        "global_momentum": global_momentum,
        "best_val_accuracy": round(float(result["best_val_accuracy"]), 6),
        "test_accuracy": round(float(metrics["accuracy"]), 6),
        "test_precision": round(float(metrics["precision"]), 6),
        "test_recall": round(float(metrics["recall"]), 6),
        "test_f1": round(float(metrics["f1"]), 6),
        "best_model_path": result["best_model_path"],
    }


def main() -> None:
    """Run a systematic parameter sweep for the full federated method."""
    parser = argparse.ArgumentParser(description="Grid search OrbitShield_FL federated parameters")
    parser.add_argument("--data_dir", default="../dataset_window")
    parser.add_argument("--output_dir", default="experiments_window/federated/full_grid")
    parser.add_argument("--num_clients", type=int, default=12)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--partition_mode", default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--beta_floor", type=float, default=0.05)
    parser.add_argument("--betas", default="0.1,0.2,0.3")
    parser.add_argument("--warmup_rounds_list", default="1,2,3")
    parser.add_argument("--global_momentum_list", default="0.1,0.2,0.3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--init_checkpoint", default="checkpoints_gru/window_gru_best.pt")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_runs", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _ = load_window_dataset(args.data_dir)

    betas = parse_float_list(args.betas)
    warmups = parse_int_list(args.warmup_rounds_list)
    momenta = parse_float_list(args.global_momentum_list)
    search_space = list(itertools.product(betas, warmups, momenta))
    if args.limit_runs is not None:
        search_space = search_space[: args.limit_runs]

    summary_rows: list[dict[str, object]] = []
    raw_results: dict[str, object] = {}
    best_run_name = None
    best_score = -1.0

    print("=" * 60)
    print("OrbitShield_FL Grid Search")
    print("=" * 60)
    print(f"Search space size: {len(search_space)}")

    for run_idx, (beta, warmup_rounds, global_momentum) in enumerate(search_space, start=1):
        run_name = format_run_name(beta, warmup_rounds, global_momentum)
        run_output_dir = os.path.join(args.output_dir, run_name)
        print(f"\n[{run_idx}/{len(search_space)}] {run_name}")
        config = FederatedConfig(
            data_dir=args.data_dir,
            output_dir=run_output_dir,
            method="full",
            num_clients=args.num_clients,
            num_planes=args.num_planes,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            partition_mode=args.partition_mode,
            dirichlet_alpha=args.dirichlet_alpha,
            beta=beta,
            beta_floor=args.beta_floor,
            lambda_s=args.lambda_s,
            rho=args.rho,
            mu=args.mu,
            global_momentum=global_momentum,
            warmup_rounds=warmup_rounds,
            device=args.device,
            init_checkpoint=args.init_checkpoint,
            hidden_dim=args.hidden_dim,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        result = run_federated_training(config)
        row = summarize_result(run_name, beta, warmup_rounds, global_momentum, result)
        summary_rows.append(row)
        raw_results[run_name] = {
            "config": {
                "beta": beta,
                "warmup_rounds": warmup_rounds,
                "global_momentum": global_momentum,
            },
            "result": result,
        }
        print(
            f"  val={row['best_val_accuracy']:.4f} "
            f"test_acc={row['test_accuracy']:.4f} "
            f"test_f1={row['test_f1']:.4f}"
        )
        if row["test_accuracy"] > best_score:
            best_score = float(row["test_accuracy"])
            best_run_name = run_name

        summary_rows.sort(key=lambda item: item["test_accuracy"], reverse=True)
        write_csv(
            os.path.join(args.output_dir, "full_grid_summary.csv"),
            summary_rows,
            [
                "variant_name",
                "method_name",
                "beta",
                "warmup_rounds",
                "global_momentum",
                "best_val_accuracy",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "best_model_path",
            ],
        )
        write_json(
            os.path.join(args.output_dir, "full_grid_results.json"),
            {
                "best_variant_name": best_run_name,
                "best_test_accuracy": best_score,
                "runs": raw_results,
            },
        )

    print(f"\nBest OrbitShield_FL variant: {best_run_name} test_acc={best_score:.4f}")


if __name__ == "__main__":
    main()
