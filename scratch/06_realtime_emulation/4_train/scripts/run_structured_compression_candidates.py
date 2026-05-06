#!/usr/bin/env python3
"""Run formal structured compression candidates with short recovery finetuning."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_DIR)

from compression_utils import benchmark_torchscript_cpu, evaluate_model, save_json
from dataset_profiles import get_dataset_profile
from experiment_utils import resolve_device, set_seed
from src.data import create_dataloaders, load_npz_data
from src.models import DSC_CBAM_GRU, StructuredDSC_CBAM_GRU, count_flops, count_parameters
from src.training import Trainer, get_optimizer, get_scheduler
from structured_compression_utils import CANDIDATES, FORMAL_CHECKPOINT, SOURCE_CFG, transfer_structured_weights


FORMAL_METRICS_JSON = os.path.join(
    PROJECT_DIR,
    "experiments",
    "comparison_formal_tuned",
    "comparison_results.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "experiments",
    "compression",
    "structured_candidates_formal_tuned",
)
TARGETS = {
    "parameter_reduction_ratio": 0.25,
    "latency_reduction_ratio": 0.20,
    "accuracy_drop_max": 0.003,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured compression candidates A1/A2/A3.")
    parser.add_argument("--dataset", choices=["cicids17"], default="cicids17")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=FORMAL_CHECKPOINT)
    parser.add_argument("--quant_engine", default="fbgemm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batch_size", type=int, default=512)
    parser.add_argument("--benchmark_steps", type=int, default=200)
    parser.add_argument("--benchmark_warmup", type=int, default=40)
    parser.add_argument("--benchmark_threads", type=int, default=4)
    return parser.parse_args()


def load_formal_reference() -> dict[str, Any]:
    with open(FORMAL_METRICS_JSON, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["dsc_cbam_gru"]["metrics"]


def make_loaders(data_dir: str, batch_size: int, eval_batch_size: int):
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(data_dir)
    train_loader, val_loader, _ = create_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    _, _, test_loader = create_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size=eval_batch_size,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def build_formal_model() -> DSC_CBAM_GRU:
    return DSC_CBAM_GRU(
        input_dim=SOURCE_CFG["input_dim"],
        num_classes=SOURCE_CFG["num_classes"],
        hidden_dim=SOURCE_CFG["hidden_dim"],
        bidirectional=SOURCE_CFG["bidirectional"],
        dropout=SOURCE_CFG["dropout"],
        conv_dim=SOURCE_CFG["conv_dim"],
        dsc_dim=SOURCE_CFG["dsc_dim"],
    )


def build_candidate_model(candidate_cfg: dict[str, Any]) -> StructuredDSC_CBAM_GRU:
    return StructuredDSC_CBAM_GRU(
        input_dim=SOURCE_CFG["input_dim"],
        num_classes=SOURCE_CFG["num_classes"],
        dropout=SOURCE_CFG["dropout"],
        bidirectional=SOURCE_CFG["bidirectional"],
        **candidate_cfg,
    )


def benchmark_cpu_pair(
    model: nn.Module,
    profile_seq_len: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float]]:
    fp32_model = copy.deepcopy(model).cpu().eval()
    fp32_latency = benchmark_torchscript_cpu(
        fp32_model,
        batch_size=args.benchmark_batch_size,
        input_dim=SOURCE_CFG["input_dim"],
        seq_len=profile_seq_len,
        threads=args.benchmark_threads,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
    )

    torch.backends.quantized.engine = args.quant_engine
    int8_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu().eval(),
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
    int8_latency = benchmark_torchscript_cpu(
        int8_model,
        batch_size=args.benchmark_batch_size,
        input_dim=SOURCE_CFG["input_dim"],
        seq_len=profile_seq_len,
        threads=args.benchmark_threads,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
    )
    return fp32_latency, int8_latency


def finetune_candidate(
    candidate_name: str,
    candidate_cfg: dict[str, Any],
    source_state: dict[str, torch.Tensor],
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    candidate_dir: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_cfg = {**SOURCE_CFG, **candidate_cfg}
    model = build_candidate_model(candidate_cfg)
    model, transferred_keys = transfer_structured_weights(source_state, model, SOURCE_CFG, target_cfg)

    transferred_checkpoint = os.path.join(candidate_dir, f"{candidate_name}_transferred.pt")
    finetuned_checkpoint = os.path.join(candidate_dir, f"{candidate_name}_finetuned.pt")
    torch.save(model.state_dict(), transferred_checkpoint)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name="adamw", lr=args.finetune_lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_name="plateau", patience=2, factor=0.5)
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.max_epochs,
        save_path=finetuned_checkpoint,
        early_stopping_patience=args.early_stopping_patience,
    )
    metrics = trainer.evaluate(test_loader)

    trained_model = build_candidate_model(candidate_cfg)
    trained_model.load_state_dict(torch.load(finetuned_checkpoint, map_location="cpu"))
    trained_model.eval()

    fp32_latency, int8_latency = benchmark_cpu_pair(trained_model, profile_seq_len=10, args=args)
    int8_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(trained_model).cpu().eval(),
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
    int8_metrics = evaluate_model(int8_model, test_loader, torch.device("cpu"))

    return {
        "candidate": candidate_name,
        "config": candidate_cfg,
        "transferred_checkpoint": transferred_checkpoint,
        "finetuned_checkpoint": finetuned_checkpoint,
        "transferred_key_count": len(transferred_keys),
        "transferred_keys": transferred_keys,
        "history": history,
        "metrics": {
            **metrics,
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        },
        "int8_metrics": int8_metrics,
        "parameter_count": int(count_parameters(trained_model)),
        "flops": int(count_flops(trained_model, input_size=(1, 10, SOURCE_CFG["input_dim"]))),
        "latency_fp32": fp32_latency,
        "latency_int8": int8_latency,
    }


def build_candidate_summary(
    result: dict[str, Any],
    baseline: dict[str, Any],
    formal_reference_metrics: dict[str, Any],
    quant_engine: str,
) -> dict[str, Any]:
    accuracy = float(result["metrics"]["accuracy"])
    f1 = float(result["metrics"]["f1"])
    parameter_count = int(result["parameter_count"])
    fp32_latency = float(result["latency_fp32"]["latency_ms_per_sample"])
    int8_latency = float(result["latency_int8"]["latency_ms_per_sample"])

    parameter_reduction_ratio = 1.0 - (parameter_count / baseline["parameter_count"])
    latency_reduction_ratio = 1.0 - (int8_latency / baseline["latency_ms_per_sample_fp32"])
    accuracy_drop_vs_formal = float(formal_reference_metrics["accuracy"] - accuracy)

    passes = {
        "parameter_reduction_ratio": bool(parameter_reduction_ratio >= TARGETS["parameter_reduction_ratio"]),
        "latency_reduction_ratio": bool(latency_reduction_ratio >= TARGETS["latency_reduction_ratio"]),
        "accuracy_drop_vs_formal": bool(accuracy_drop_vs_formal <= TARGETS["accuracy_drop_max"]),
    }

    return {
        "candidate": result["candidate"],
        "config": result["config"],
        "accuracy": accuracy,
        "f1": f1,
        "int8_accuracy": float(result["int8_metrics"]["accuracy"]),
        "int8_f1": float(result["int8_metrics"]["f1"]),
        "parameter_count": parameter_count,
        "flops": int(result["flops"]),
        "fp32_cpu_latency_ms": fp32_latency,
        "int8_cpu_latency_ms": int8_latency,
        "parameter_reduction_ratio": float(parameter_reduction_ratio),
        "latency_reduction_ratio": float(latency_reduction_ratio),
        "accuracy_drop_vs_formal": accuracy_drop_vs_formal,
        "quant_engine": quant_engine,
        "baseline_formal": baseline,
        "formal_reference_metrics": formal_reference_metrics,
        "passes_thresholds": passes,
        "meets_all_targets": bool(all(passes.values())),
        "artifacts": {
            "transferred_checkpoint": result["transferred_checkpoint"],
            "finetuned_checkpoint": result["finetuned_checkpoint"],
            "transferred_key_count": result["transferred_key_count"],
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    profile = get_dataset_profile(args.dataset)
    data_dir = args.data_dir or profile.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_DIR, data_dir)

    if args.quant_engine not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Unsupported quantized engine: {args.quant_engine}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    train_loader, val_loader, test_loader = make_loaders(data_dir, args.batch_size, args.eval_batch_size)
    formal_reference_metrics = load_formal_reference()

    source_state = torch.load(args.checkpoint, map_location="cpu")

    formal_model = build_formal_model()
    formal_model.load_state_dict(source_state)
    formal_model.eval()
    baseline_fp32_metrics = evaluate_model(copy.deepcopy(formal_model).to(device).eval(), test_loader, device)
    baseline_fp32_latency, baseline_int8_latency = benchmark_cpu_pair(formal_model, profile.seq_len, args)
    baseline = {
        "checkpoint": args.checkpoint,
        "accuracy": float(baseline_fp32_metrics["accuracy"]),
        "f1": float(baseline_fp32_metrics["f1"]),
        "parameter_count": int(count_parameters(formal_model)),
        "flops": int(count_flops(formal_model, input_size=(1, profile.seq_len, profile.input_dim))),
        "latency_ms_per_sample_fp32": float(baseline_fp32_latency["latency_ms_per_sample"]),
        "latency_ms_per_sample_int8": float(baseline_int8_latency["latency_ms_per_sample"]),
        "quant_engine": args.quant_engine,
    }

    aggregate = {
        "formal_tuned_baseline": baseline,
        "formal_reference_metrics": formal_reference_metrics,
        "targets": TARGETS,
        "candidates": {},
    }

    for candidate_name, candidate_cfg in CANDIDATES.items():
        print(f"\n=== Running {candidate_name} ===")
        candidate_dir = os.path.join(args.output_dir, candidate_name)
        os.makedirs(candidate_dir, exist_ok=True)
        result = finetune_candidate(
            candidate_name,
            candidate_cfg,
            source_state,
            train_loader,
            val_loader,
            test_loader,
            device,
            candidate_dir,
            args,
        )
        summary = build_candidate_summary(result, baseline, formal_reference_metrics, args.quant_engine)
        candidate_summary_path = os.path.join(candidate_dir, "candidate_summary.json")
        save_json(candidate_summary_path, summary)
        aggregate["candidates"][candidate_name] = {
            **summary,
            "summary_path": candidate_summary_path,
        }
        print(
            f"{candidate_name}: acc={summary['accuracy']:.6f}, f1={summary['f1']:.6f}, "
            f"params={summary['parameter_count']}, fp32_cpu_ms={summary['fp32_cpu_latency_ms']:.6f}, "
            f"int8_cpu_ms={summary['int8_cpu_latency_ms']:.6f}, meets_all={summary['meets_all_targets']}"
        )

    aggregate_path = os.path.join(args.output_dir, "all_candidates_summary.json")
    save_json(aggregate_path, aggregate)
    print(f"\nSaved aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
