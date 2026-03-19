#!/usr/bin/env python3
"""Train the OrbitShield_FL federated threat predictor on CICIDS17 or STI."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from OrbitShield_FL import FederatedConfig, METHOD_PRESETS, run_federated_training
from dataset_profiles import get_dataset_profile


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the federated trainer."""
    parser = argparse.ArgumentParser(description="OrbitShield_FL federated DSC-CBAM-GRU training")

    common = parser.add_argument_group("common")
    common.add_argument("--dataset", choices=["cicids17", "sti"], default="cicids17")
    common.add_argument("--method", default="full", choices=["single", "fedavg", "intra_only", "intra_gossip", "full"])
    common.add_argument("--rounds", type=int, default=20)
    common.add_argument("--local_epochs", type=int, default=1)
    common.add_argument("--batch_size", type=int, default=512)
    common.add_argument("--num_clients", type=int, default=12)
    common.add_argument("--num_planes", type=int, default=3)
    common.add_argument("--device", default=None)
    common.add_argument("--output_dir", default=None)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--full_eval",
        action="store_true",
        help="Use full validation/test splits and full local passes. Recommended only for STI final runs.",
    )
    common.add_argument(
        "--from_scratch",
        action="store_true",
        help="Disable warm start checkpoint and train federated model from scratch.",
    )

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument("--data_dir", default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--partition_mode", default="dirichlet", help=argparse.SUPPRESS)
    advanced.add_argument("--dirichlet_alpha", type=float, default=0.3, help=argparse.SUPPRESS)
    advanced.add_argument("--lr", type=float, default=1e-3, help=argparse.SUPPRESS)
    advanced.add_argument("--weight_decay", type=float, default=1e-2, help=argparse.SUPPRESS)
    advanced.add_argument("--init_checkpoint", default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--hidden_dim", type=int, default=64, help=argparse.SUPPRESS)
    advanced.add_argument("--bidirectional", action="store_true", default=False, help=argparse.SUPPRESS)
    advanced.add_argument("--dropout", type=float, default=0.3, help=argparse.SUPPRESS)
    advanced.add_argument("--input_dim", type=int, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--num_classes", type=int, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--max_samples", type=int, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--max_local_batches", type=int, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--eval_subset_size", type=int, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--test_subset_size", type=int, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--eval_every", type=int, default=1, help=argparse.SUPPRESS)
    advanced.add_argument("--test_every", type=int, default=0, help=argparse.SUPPRESS)
    advanced.add_argument("--beta", type=float, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--beta_floor", type=float, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--lambda_s", type=float, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--rho", type=float, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--mu", type=float, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--global_momentum", type=float, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--warmup_rounds", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_output_dir(dataset_name: str, method: str, explicit_output_dir: str | None) -> str:
    """Resolve the default output directory for a dataset/method pair."""
    if explicit_output_dir:
        return explicit_output_dir
    if dataset_name == "cicids17" and method == "full":
        return "experiments/OrbitShield_FL/cicids17"
    if dataset_name == "sti" and method == "full":
        return "experiments/OrbitShield_FL/sti"
    suffix = "OrbitShield_FL" if method == "full" else method
    return f"experiments/OrbitShield_FL/{suffix}_{dataset_name}"


def resolve_runtime_defaults(args: argparse.Namespace, dataset_name: str) -> dict[str, object]:
    """Resolve dataset-specific runtime defaults while keeping CLI surface small."""
    eval_subset_size = args.eval_subset_size
    test_subset_size = args.test_subset_size
    max_local_batches = args.max_local_batches
    test_every = args.test_every

    if eval_subset_size is not None and eval_subset_size <= 0:
        eval_subset_size = None
    if test_subset_size is not None and test_subset_size <= 0:
        test_subset_size = None
    if max_local_batches is not None and max_local_batches <= 0:
        max_local_batches = None

    if dataset_name == "sti":
        if args.full_eval:
            eval_subset_size = None
            test_subset_size = None
            max_local_batches = None
            test_every = 1
        else:
            if eval_subset_size is None:
                eval_subset_size = 50000
            if max_local_batches is None:
                max_local_batches = 128
            test_every = 0 if args.test_every == 0 else args.test_every
    return {
        "eval_subset_size": eval_subset_size,
        "test_subset_size": test_subset_size,
        "max_local_batches": max_local_batches,
        "test_every": test_every,
    }


def resolve_method_hparams(args: argparse.Namespace) -> dict[str, object]:
    """Resolve federated method hyperparameters from formal presets plus optional overrides."""
    params = METHOD_PRESETS[args.method].copy()
    overrides = {
        "beta": args.beta,
        "beta_floor": args.beta_floor,
        "lambda_s": args.lambda_s,
        "rho": args.rho,
        "mu": args.mu,
        "global_momentum": args.global_momentum,
        "warmup_rounds": args.warmup_rounds,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    return params


def main() -> None:
    """Run federated training from CLI arguments."""
    args = parse_args()
    profile = get_dataset_profile(args.dataset)
    runtime_defaults = resolve_runtime_defaults(args, profile.name)
    method_hparams = resolve_method_hparams(args)

    config = FederatedConfig(
        dataset=args.dataset,
        data_dir=args.data_dir or profile.data_dir,
        output_dir=resolve_output_dir(profile.name, args.method, args.output_dir),
        method=args.method,
        num_clients=args.num_clients,
        num_planes=args.num_planes,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        max_local_batches=runtime_defaults["max_local_batches"],
        batch_size=args.batch_size,
        input_dim=args.input_dim or profile.input_dim,
        num_classes=args.num_classes or profile.num_classes,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        init_checkpoint=None if args.from_scratch else (args.init_checkpoint if args.init_checkpoint is not None else profile.init_checkpoint),
        max_samples=args.max_samples,
        eval_subset_size=runtime_defaults["eval_subset_size"],
        test_subset_size=runtime_defaults["test_subset_size"],
        eval_every=args.eval_every,
        test_every=runtime_defaults["test_every"],
        partition_mode=args.partition_mode,
        dirichlet_alpha=args.dirichlet_alpha,
        beta=float(method_hparams["beta"]),
        beta_floor=float(method_hparams["beta_floor"]),
        lambda_s=float(method_hparams["lambda_s"]),
        rho=float(method_hparams["rho"]),
        mu=float(method_hparams["mu"]),
        global_momentum=float(method_hparams["global_momentum"]),
        warmup_rounds=int(method_hparams["warmup_rounds"]),
        class_names=profile.class_names,
        seed=args.seed,
        device=args.device,
    )

    result = run_federated_training(config)
    print("=" * 60)
    print("OrbitShield_FL Training Finished")
    print("=" * 60)
    print(f"Dataset     : {profile.name}")
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
