#!/usr/bin/env python3
"""Train OrbitShield_FL with an ns-3-driven communication topology backend."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset_profiles import get_dataset_profile
from OrbitShield_FL import FederatedConfig, run_federated_training
from OrbitShield_FL.ns3_bridge import load_ns3_trace_bundle, run_federated_constellation
from train_federated import resolve_method_hparams, resolve_runtime_defaults


DEFAULT_NS3_BINARY = (
    "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/06_realtime_emulation/"
    "ns3.46.1-federated_constellation-optimized"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ns-3-driven federated trainer."""
    parser = argparse.ArgumentParser(description="OrbitShield_FL federated training with ns-3 topology traces")

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

    ns3_group = parser.add_argument_group("ns3")
    ns3_group.add_argument("--trace_dir", default=None, help="Use an existing ns-3 trace directory")
    ns3_group.add_argument("--generate_trace", action="store_true", help="Generate a fresh trace before training")
    ns3_group.add_argument("--ns3_binary", default=DEFAULT_NS3_BINARY)
    ns3_group.add_argument("--trace_output_dir", default=None)
    ns3_group.add_argument("--round_duration", type=float, default=30.0)
    ns3_group.add_argument("--contact_period", type=int, default=4)
    ns3_group.add_argument("--contact_duration_rounds", type=int, default=2)

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


def resolve_output_dir(dataset_name: str, explicit_output_dir: str | None) -> str:
    """Resolve the default output directory for ns-3-driven training."""
    if explicit_output_dir:
        return explicit_output_dir
    if dataset_name == "cicids17":
        return "experiments/OrbitShield_FL_ns3/cicids17"
    return f"experiments/OrbitShield_FL_ns3/{dataset_name}"


def resolve_default_trace_output_dir(dataset_name: str) -> str:
    """Resolve the formal default trace directory for ns-3-driven training."""
    if dataset_name == "cicids17":
        return "experiments/OrbitShield_FL_ns3/cicids17_trace"
    return f"experiments/OrbitShield_FL_ns3/{dataset_name}_trace"


def resolve_trace_dir(args: argparse.Namespace, output_dir: str) -> str:
    """Resolve or generate the ns-3 trace directory used by this run."""
    if args.trace_dir:
        bundle = load_ns3_trace_bundle(args.trace_dir)
        if len(bundle.rounds) < args.rounds:
            raise ValueError(
                f"Trace directory {bundle.trace_dir} contains only {len(bundle.rounds)} rounds, "
                f"but {args.rounds} are required"
            )
        return str(bundle.trace_dir)

    trace_dir = args.trace_output_dir or resolve_default_trace_output_dir(args.dataset)
    if args.generate_trace or not Path(trace_dir).exists():
        bundle = run_federated_constellation(
            binary_path=args.ns3_binary,
            output_dir=trace_dir,
            num_planes=args.num_planes,
            sats_per_plane=args.num_clients // args.num_planes,
            rounds=args.rounds,
            round_duration=args.round_duration,
            seed=args.seed,
            extra_args=[
                f"--contact-period={args.contact_period}",
                f"--contact-duration-rounds={args.contact_duration_rounds}",
            ],
        )
        return str(bundle.trace_dir)

    bundle = load_ns3_trace_bundle(trace_dir)
    if len(bundle.rounds) < args.rounds:
        raise ValueError(
            f"Trace directory {bundle.trace_dir} contains only {len(bundle.rounds)} rounds, "
            f"but {args.rounds} are required"
        )
    return str(bundle.trace_dir)


def main() -> None:
    """Run federated training using ns-3-generated topology traces."""
    args = parse_args()
    profile = get_dataset_profile(args.dataset)
    runtime_defaults = resolve_runtime_defaults(args, profile.name)
    method_hparams = resolve_method_hparams(args)
    output_dir = resolve_output_dir(profile.name, args.output_dir)
    trace_dir = resolve_trace_dir(args, output_dir)

    config = FederatedConfig(
        dataset=args.dataset,
        data_dir=args.data_dir or profile.data_dir,
        output_dir=output_dir,
        topology_backend="ns3",
        ns3_trace_dir=trace_dir,
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
    print("OrbitShield_FL ns-3 Training Finished")
    print("=" * 60)
    print(f"Dataset     : {profile.name}")
    print(f"Trace dir   : {trace_dir}")
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
