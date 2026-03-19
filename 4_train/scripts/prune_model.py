#!/usr/bin/env python3
"""Post-training pruning for the DSC-CBAM-GRU model."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compression_utils import (
    benchmark_inference,
    build_model,
    checkpoint_size_mb,
    create_test_loader,
    evaluate_model,
    load_model_checkpoint,
    save_json,
    summarize_dense_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply global unstructured pruning to DSC-CBAM-GRU")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression/pruning")
    parser.add_argument("--amount", type=float, default=0.30, help="Global pruning ratio in [0, 1]")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batches", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def collect_prunable_parameters(model) -> list[tuple[torch.nn.Module, str]]:
    """Collect Conv1d, Linear and GRU weights for global pruning."""
    parameters: list[tuple[torch.nn.Module, str]] = [
        (model.conv, "weight"),
        (model.dsc.dw, "weight"),
        (model.dsc.pw, "weight"),
        (model.cbam.channel_attention.fc[0], "weight"),
        (model.cbam.channel_attention.fc[2], "weight"),
        (model.cbam.spatial_attention.conv, "weight"),
        (model.fc[0], "weight"),
        (model.fc[3], "weight"),
        (model.gru, "weight_ih_l0"),
        (model.gru, "weight_hh_l0"),
    ]

    if model.bidirectional:
        parameters.extend(
            [
                (model.gru, "weight_ih_l0_reverse"),
                (model.gru, "weight_hh_l0_reverse"),
            ]
        )

    return parameters


def remove_pruning_reparametrization(parameters: list[tuple[torch.nn.Module, str]]) -> None:
    """Make pruning permanent so the saved checkpoint is a normal state_dict."""
    for module, parameter_name in parameters:
        prune.remove(module, parameter_name)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(args.output_dir, exist_ok=True)
    test_loader = create_test_loader(args.data_dir, args.batch_size)

    baseline_model = build_model(args)
    baseline_model = load_model_checkpoint(baseline_model, args.checkpoint, device)
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    baseline_latency = benchmark_inference(
        baseline_model,
        test_loader,
        device,
        num_batches=args.benchmark_batches,
    )

    pruned_model = build_model(args)
    pruned_model = load_model_checkpoint(pruned_model, args.checkpoint, device)
    prunable_parameters = collect_prunable_parameters(pruned_model)
    prune.global_unstructured(
        prunable_parameters,
        pruning_method=prune.L1Unstructured,
        amount=args.amount,
    )
    remove_pruning_reparametrization(prunable_parameters)
    pruned_model.eval()

    output_checkpoint = os.path.join(
        args.output_dir,
        f"cicids17_gru_pruned_{int(args.amount * 100):02d}.pt",
    )
    torch.save(pruned_model.state_dict(), output_checkpoint)

    pruned_metrics = evaluate_model(pruned_model, test_loader, device)
    pruned_latency = benchmark_inference(
        pruned_model,
        test_loader,
        device,
        num_batches=args.benchmark_batches,
    )

    baseline_summary = summarize_dense_model(baseline_model)
    pruned_summary = summarize_dense_model(pruned_model)

    summary = {
        "compression": "global_unstructured_pruning",
        "pruning_amount": args.amount,
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": output_checkpoint,
        "baseline": {
            **baseline_metrics,
            **baseline_latency,
            **baseline_summary,
            "checkpoint_size_mb": checkpoint_size_mb(args.checkpoint),
        },
        "pruned": {
            **pruned_metrics,
            **pruned_latency,
            **pruned_summary,
            "checkpoint_size_mb": checkpoint_size_mb(output_checkpoint),
        },
    }

    output_json = os.path.join(
        args.output_dir,
        f"cicids17_gru_pruned_{int(args.amount * 100):02d}_summary.json",
    )
    save_json(output_json, summary)

    print("=" * 60)
    print("Pruning Summary")
    print("=" * 60)
    print(f"Baseline accuracy : {baseline_metrics['accuracy']:.4f}")
    print(f"Pruned accuracy   : {pruned_metrics['accuracy']:.4f}")
    print(f"Baseline F1       : {baseline_metrics['f1']:.4f}")
    print(f"Pruned F1         : {pruned_metrics['f1']:.4f}")
    print(f"Pruned sparsity   : {pruned_summary['sparsity']:.4f}")
    print(f"Baseline size MB  : {summary['baseline']['checkpoint_size_mb']:.4f}")
    print(f"Pruned size MB    : {summary['pruned']['checkpoint_size_mb']:.4f}")
    print(f"Saved checkpoint  : {output_checkpoint}")
    print(f"Saved summary     : {output_json}")


if __name__ == "__main__":
    main()
