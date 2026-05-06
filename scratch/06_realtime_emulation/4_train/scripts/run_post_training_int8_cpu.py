#!/usr/bin/env python3
"""Formal post-training pruning + INT8 CPU experiment based on an existing checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compression_utils import (
    apply_targeted_pruning,
    benchmark_torchscript_cpu,
    build_model,
    checkpoint_size_mb,
    count_nonzero_weights,
    create_test_loader,
    evaluate_model,
    load_model_checkpoint,
    save_json,
    summarize_dense_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training pruning + INT8 experiment on CPU.")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru_formal_tuned/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression/post_training_int8_cpu_formal_tuned")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batch_size", type=int, default=512)
    parser.add_argument("--benchmark_steps", type=int, default=200)
    parser.add_argument("--benchmark_warmup", type=int, default=40)
    parser.add_argument("--benchmark_threads", type=int, default=4)
    parser.add_argument("--quant_engine", default="fbgemm")
    parser.add_argument("--gru_prune_amount", type=float, default=0.35)
    parser.add_argument("--fc_prune_amount", type=float, default=0.20)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--conv_dim", type=int, default=16)
    parser.add_argument("--dsc_dim", type=int, default=48)
    parser.add_argument("--seq_len", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    if args.quant_engine not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Unsupported quantized engine: {args.quant_engine}")

    os.makedirs(args.output_dir, exist_ok=True)
    test_loader = create_test_loader(args.data_dir, args.eval_batch_size)

    baseline_model = load_model_checkpoint(build_model(args), args.checkpoint, device)
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    baseline_latency = benchmark_torchscript_cpu(
        baseline_model,
        batch_size=args.benchmark_batch_size,
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        threads=args.benchmark_threads,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
    )
    baseline_summary = summarize_dense_model(baseline_model)

    compressed_model = load_model_checkpoint(build_model(args), args.checkpoint, device)
    apply_targeted_pruning(
        compressed_model,
        gru_amount=args.gru_prune_amount,
        fc_amount=args.fc_prune_amount,
    )
    total_elements, nonzero_elements = count_nonzero_weights(compressed_model)

    torch.backends.quantized.engine = args.quant_engine
    quantized_model = torch.quantization.quantize_dynamic(
        compressed_model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
    quantized_model.eval()

    output_checkpoint = os.path.join(args.output_dir, "cicids17_gru_post_training_int8.pt")
    torch.save(
        {
            "compression": "post_training_pruning_plus_dynamic_int8",
            "config": {
                "gru_prune_amount": args.gru_prune_amount,
                "fc_prune_amount": args.fc_prune_amount,
                "quant_engine": args.quant_engine,
                "hidden_dim": args.hidden_dim,
                "bidirectional": args.bidirectional,
                "dropout": args.dropout,
            },
            "state_dict": quantized_model.state_dict(),
        },
        output_checkpoint,
    )

    quantized_metrics = evaluate_model(quantized_model, test_loader, device)
    quantized_latency = benchmark_torchscript_cpu(
        quantized_model,
        batch_size=args.benchmark_batch_size,
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        threads=args.benchmark_threads,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
    )

    parameter_reduction_ratio = float(1.0 - (nonzero_elements / total_elements if total_elements else 0.0))
    latency_reduction_ratio = float(
        1.0
        - (
            quantized_latency["latency_ms_per_sample"]
            / baseline_latency["latency_ms_per_sample"]
            if baseline_latency["latency_ms_per_sample"] > 0
            else 0.0
        )
    )

    summary = {
        "experiment": "post_training_int8_cpu",
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": output_checkpoint,
        "baseline": {
            **baseline_metrics,
            **baseline_latency,
            **baseline_summary,
            "dtype": "fp32",
            "device": "cpu",
            "checkpoint_size_mb": checkpoint_size_mb(args.checkpoint),
        },
        "compressed": {
            **quantized_metrics,
            **quantized_latency,
            "dtype": "int8_dynamic",
            "device": "cpu",
            "effective_parameter_count": int(nonzero_elements),
            "parameter_reduction_ratio": parameter_reduction_ratio,
            "checkpoint_size_mb": checkpoint_size_mb(output_checkpoint),
        },
        "config": {
            "gru_prune_amount": args.gru_prune_amount,
            "fc_prune_amount": args.fc_prune_amount,
            "quant_engine": args.quant_engine,
            "benchmark_batch_size": args.benchmark_batch_size,
            "benchmark_steps": args.benchmark_steps,
            "benchmark_warmup": args.benchmark_warmup,
            "benchmark_threads": args.benchmark_threads,
            "benchmark_backend": "torchscript_cpu",
        },
        "latency_reduction_ratio": latency_reduction_ratio,
        "meets_target": {
            "accuracy_gte_0_95": bool(quantized_metrics["accuracy"] >= 0.95),
            "parameter_reduction_gte_0_25": bool(parameter_reduction_ratio >= 0.25),
            "latency_reduction_gte_0_20": bool(latency_reduction_ratio >= 0.20),
        },
    }

    output_json = os.path.join(args.output_dir, "cicids17_gru_post_training_int8_summary.json")
    save_json(output_json, summary)
    print(f"Saved checkpoint: {output_checkpoint}")
    print(f"Saved summary   : {output_json}")


if __name__ == "__main__":
    main()
