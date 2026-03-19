#!/usr/bin/env python3
"""Post-training dynamic quantization for the DSC-CBAM-GRU model."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

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
    parser = argparse.ArgumentParser(description="Apply dynamic quantization to DSC-CBAM-GRU")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression/quantization")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batches", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

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
    baseline_summary = summarize_dense_model(baseline_model)

    quantized_model = build_model(args)
    quantized_model = load_model_checkpoint(quantized_model, args.checkpoint, device)
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
    quantized_model.eval()

    output_checkpoint = os.path.join(args.output_dir, "cicids17_gru_dynamic_int8.pt")
    torch.save(
        {
            "quantized_dynamic": True,
            "config": {
                "input_dim": args.input_dim,
                "num_classes": args.num_classes,
                "hidden_dim": args.hidden_dim,
                "bidirectional": args.bidirectional,
                "dropout": args.dropout,
            },
            "state_dict": quantized_model.state_dict(),
        },
        output_checkpoint,
    )

    quantized_metrics = evaluate_model(quantized_model, test_loader, device)
    quantized_latency = benchmark_inference(
        quantized_model,
        test_loader,
        device,
        num_batches=args.benchmark_batches,
    )

    summary = {
        "compression": "dynamic_quantization_int8",
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": output_checkpoint,
        "baseline": {
            **baseline_metrics,
            **baseline_latency,
            **baseline_summary,
            "checkpoint_size_mb": checkpoint_size_mb(args.checkpoint),
        },
        "quantized": {
            **quantized_metrics,
            **quantized_latency,
            "parameter_count": int(baseline_summary["parameter_count"]),
            "checkpoint_size_mb": checkpoint_size_mb(output_checkpoint),
        },
    }

    output_json = os.path.join(args.output_dir, "cicids17_gru_dynamic_int8_summary.json")
    save_json(output_json, summary)

    print("=" * 60)
    print("Quantization Summary")
    print("=" * 60)
    print(f"Baseline accuracy : {baseline_metrics['accuracy']:.4f}")
    print(f"Quantized accuracy: {quantized_metrics['accuracy']:.4f}")
    print(f"Baseline F1       : {baseline_metrics['f1']:.4f}")
    print(f"Quantized F1      : {quantized_metrics['f1']:.4f}")
    print(f"Baseline size MB  : {summary['baseline']['checkpoint_size_mb']:.4f}")
    print(f"Quantized size MB : {summary['quantized']['checkpoint_size_mb']:.4f}")
    print(f"Saved checkpoint  : {output_checkpoint}")
    print(f"Saved summary     : {output_json}")


if __name__ == "__main__":
    main()
