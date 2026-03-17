#!/usr/bin/env python3
"""Train the OrbitShield_FL federated threat predictor on dataset_window."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from federated import FederatedConfig, run_federated_training


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the federated trainer."""
    parser = argparse.ArgumentParser(description="OrbitShield_FL federated DSC-CBAM-GRU training on dataset_window")
    parser.add_argument("--data_dir", default="../dataset_window")
    parser.add_argument("--num_clients", type=int, default=12)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--partition_mode", default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--beta_floor", type=float, default=0.05)
    parser.add_argument("--lambda_s", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--global_momentum", type=float, default=0.1)
    parser.add_argument("--warmup_rounds", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="experiments_window/federated/OrbitShield_FL")
    parser.add_argument("--method", default="full", choices=["single", "fedavg", "intra_only", "intra_gossip", "full"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--init_checkpoint", default="checkpoints_gru/window_gru_best.pt")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run federated training from CLI arguments."""
    args = parse_args()
    config = FederatedConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        num_clients=args.num_clients,
        num_planes=args.num_planes,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        partition_mode=args.partition_mode,
        dirichlet_alpha=args.dirichlet_alpha,
        beta=args.beta,
        beta_floor=args.beta_floor,
        lambda_s=args.lambda_s,
        rho=args.rho,
        mu=args.mu,
        global_momentum=args.global_momentum,
        warmup_rounds=args.warmup_rounds,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        init_checkpoint=args.init_checkpoint,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        seed=args.seed,
    )
    result = run_federated_training(config)
    print("=" * 60)
    print("OrbitShield_FL Training Finished")
    print("=" * 60)
    print(f"Method label: {'OrbitShield_FL' if args.method == 'full' else args.method}")
    print(f"Best model: {result['best_model_path']}")
    print(f"Best val accuracy: {result['best_val_accuracy']:.4f}")
    final_metrics = result["final_test_metrics"]
    print(f"Test Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Test Precision: {final_metrics['precision']:.4f}")
    print(f"Test Recall   : {final_metrics['recall']:.4f}")
    print(f"Test F1       : {final_metrics['f1']:.4f}")
    print(final_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
