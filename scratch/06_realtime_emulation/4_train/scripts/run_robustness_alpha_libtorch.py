#!/usr/bin/env python3
"""Robustness experiment: sweep Dirichlet alpha for Level 4B (ns-3 + libtorch)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/lithic/final/ns3-gpu-venv/bin/python"

ALPHAS = [0.05, 0.1, 0.3, 0.5, 1.0, 5.0]
ROUNDS = 10
BATCH_SIZE = 512
DEVICE = "cuda"
DATASET = "cicids17"
INIT_CHECKPOINT = "checkpoints_gru/cicids17_gru_best.pt"
OUTPUT_BASE = TRAIN_ROOT / "experiments" / "robustness_libtorch" / "alpha_sweep"
OUTPUT_BASE = TRAIN_ROOT / "experiments" / "OrbitShield_FL_ns3_libtorch" / "alpha_sweep"
EXPORT_BASE = TRAIN_ROOT / "libtorch_data" / DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Dirichlet alpha for Level 4B libtorch runtime")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=ALPHAS,
        help="Alpha values to run. Defaults to the full sweep.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected alphas even if summary.json already exists.",
    )
    return parser.parse_args()


def run_one(alpha: float, force: bool = False) -> dict:
    alpha_tag = str(alpha).replace(".", "p")
    export_dir = EXPORT_BASE
    output_dir = OUTPUT_BASE / f"alpha_{alpha_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] alpha={alpha} already done, loading existing results.")
        return json.loads(summary_path.read_text())

    print(f"\n{'='*60}")
    print(f"Running Level 4B: alpha={alpha}")
    print(f"  export_dir : {export_dir}")
    print(f"  output_dir : {output_dir}")
    print(f"{'='*60}")

    cmd = [
        PYTHON,
        str(TRAIN_ROOT / "scripts" / "train_federated_ns3_libtorch.py"),
        "--dataset", DATASET,
        "--rounds", str(ROUNDS),
        "--local_epochs", "1",
        "--batch_size", str(BATCH_SIZE),
        "--device", DEVICE,
        "--partition_mode", "dirichlet",
        "--dirichlet_alpha", str(alpha),
        "--force_export",
        "--export_dir", str(export_dir),
        "--output_dir", str(output_dir),
        "--init_checkpoint", INIT_CHECKPOINT,
    ]

    env = os.environ.copy()
    torch_lib = "/home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/lib"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{existing}".rstrip(":")

    result = subprocess.run(cmd, cwd=TRAIN_ROOT, env=env)
    if result.returncode != 0:
        print(f"[ERROR] alpha={alpha} failed with returncode {result.returncode}")
        return {}

    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def main() -> None:
    args = parse_args()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    EXPORT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = []

    for alpha in args.alphas:
        summary = run_one(alpha, force=args.force)
        if summary:
            row = {
                "alpha": alpha,
                "best_val_accuracy": summary.get("best_val_accuracy", float("nan")),
                "test_accuracy": summary.get("test_accuracy", float("nan")),
                "test_f1": summary.get("test_f1", float("nan")),
            }
            all_results.append(row)
            print(f"  alpha={alpha:5.2f}  val={row['best_val_accuracy']:.4f}  test_acc={row['test_accuracy']:.4f}  f1={row['test_f1']:.4f}")

    csv_path = OUTPUT_BASE / "alpha_sweep_summary.csv"
    with open(csv_path, "w") as f:
        f.write("alpha,best_val_accuracy,test_accuracy,test_f1\n")
        for row in all_results:
            f.write(f"{row['alpha']},{row['best_val_accuracy']:.6f},{row['test_accuracy']:.6f},{row['test_f1']:.6f}\n")

    print(f"\nSummary written to: {csv_path}")


if __name__ == "__main__":
    main()
