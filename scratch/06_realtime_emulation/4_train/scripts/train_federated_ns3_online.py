#!/usr/bin/env python3
"""Train OrbitShield_FL with Level 3 online ns-3 co-simulation."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset_profiles import get_dataset_profile
from OrbitShield_FL import FederatedConfig, run_online_federated_training
from train_federated import resolve_method_hparams, resolve_runtime_defaults


DEFAULT_NS3_BINARY = (
    "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/06_realtime_emulation/"
    "ns3.46.1-federated_constellation-optimized"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Level 3 online co-simulation."""
    parser = argparse.ArgumentParser(description="OrbitShield_FL Level 3 online ns-3 co-simulation training")

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
    common.add_argument("--full_eval", action="store_true")
    common.add_argument("--from_scratch", action="store_true")

    online = parser.add_argument_group("online_ns3")
    online.add_argument("--ns3_binary", default=DEFAULT_NS3_BINARY)
    online.add_argument("--round_duration", type=float, default=30.0)
    online.add_argument("--contact_period", type=int, default=4)
    online.add_argument("--contact_duration_rounds", type=int, default=2)
    online.add_argument("--reuse_round_trace", action="store_true", help="Reuse generated per-round traces if they already exist")

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
    advanced.add_argument("--conv_dim", type=int, default=32, help=argparse.SUPPRESS)
    advanced.add_argument("--dsc_dim", type=int, default=64, help=argparse.SUPPRESS)
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


def resolve_output_dir(dataset_name: str, explicit_output_dir: str | None) -> str:
    """Resolve the formal default output directory for Level 3 training."""
    if explicit_output_dir:
        return explicit_output_dir
    if dataset_name == "cicids17":
        return "experiments/OrbitShield_FL_ns3_online/cicids17"
    return f"experiments/OrbitShield_FL_ns3_online/{dataset_name}"


def main() -> None:
    """Run Level 3 online co-simulation federated training."""
    args = parse_args()
    profile = get_dataset_profile(args.dataset)
    runtime_defaults = resolve_runtime_defaults(args, profile.name)
    method_hparams = resolve_method_hparams(args)
    output_dir = resolve_output_dir(profile.name, args.output_dir)

    config = FederatedConfig(
        dataset=args.dataset,
        data_dir=args.data_dir or profile.data_dir,
        output_dir=output_dir,
        topology_backend="ns3_online",
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
        conv_dim=args.conv_dim,
        dsc_dim=args.dsc_dim,
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
        ns3_binary=args.ns3_binary,
        ns3_round_duration=args.round_duration,
        ns3_force_regenerate=not args.reuse_round_trace,
        inter_plane_contact_period=args.contact_period,
        inter_plane_contact_duration=args.contact_duration_rounds,
    )

    result = run_online_federated_training(config)
    print("=" * 60)
    print("OrbitShield_FL Level 3 Online Co-Simulation Finished")
    print("=" * 60)
    print(f"Dataset     : {profile.name}")
    print(f"Output dir  : {output_dir}")
    print(f"Best model  : {result['best_model_path']}")
    print(f"Best val accuracy: {result['best_val_accuracy']:.4f}")
    final_metrics = result["final_test_metrics"]
    print(f"Test Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Test Precision: {final_metrics['precision']:.4f}")
    print(f"Test Recall   : {final_metrics['recall']:.4f}")
    print(f"Test F1       : {final_metrics['f1']:.4f}")
    print(final_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
