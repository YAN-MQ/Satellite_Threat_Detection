#!/usr/bin/env python3
"""Shared helpers for post-training compression experiments."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.data import create_dataloaders, load_npz_data
from src.models import DSC_CBAM_GRU, count_parameters


def build_model(args: Any) -> DSC_CBAM_GRU:
    """Instantiate a DSC-CBAM-GRU using the CLI shape/config arguments."""
    return DSC_CBAM_GRU(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    )


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load a state_dict checkpoint into a model."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def create_test_loader(data_dir: str, batch_size: int):
    """Create a test loader using the existing dataset pipeline."""
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(data_dir)
    _, _, test_loader = create_dataloaders(
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
    return test_loader


def evaluate_model(model: torch.nn.Module, data_loader, device: torch.device) -> dict[str, Any]:
    """Evaluate a model on a loader and return standard classification metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="weighted")),
        "recall": float(recall_score(all_labels, all_preds, average="weighted")),
        "f1": float(f1_score(all_labels, all_preds, average="weighted")),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


def benchmark_inference(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> dict[str, float]:
    """Measure per-batch and per-sample inference latency."""
    model.eval()
    measured_batches = 0
    measured_samples = 0
    total_seconds = 0.0

    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(data_loader):
            features = features.to(device)

            start = time.perf_counter()
            _ = model(features)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            if batch_idx >= warmup_batches:
                total_seconds += elapsed
                measured_batches += 1
                measured_samples += features.size(0)

            if measured_batches >= num_batches:
                break

    if measured_batches == 0 or measured_samples == 0:
        return {
            "inference_time_sec": 0.0,
            "latency_ms_per_batch": 0.0,
            "latency_ms_per_sample": 0.0,
            "throughput_samples_per_sec": 0.0,
        }

    return {
        "inference_time_sec": float(total_seconds),
        "latency_ms_per_batch": float(total_seconds * 1000.0 / measured_batches),
        "latency_ms_per_sample": float(total_seconds * 1000.0 / measured_samples),
        "throughput_samples_per_sec": float(measured_samples / total_seconds),
    }


def checkpoint_size_mb(path: str) -> float:
    """Return checkpoint size in megabytes."""
    return float(os.path.getsize(path) / (1024.0 * 1024.0))


def count_nonzero_weights(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and nonzero parameters from the current dense tensors."""
    total = 0
    nonzero = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        total += parameter.numel()
        nonzero += int(torch.count_nonzero(parameter).item())
    return total, nonzero


def save_json(path: str, payload: dict[str, Any]) -> None:
    """Save a JSON file with stable formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def summarize_dense_model(model: torch.nn.Module) -> dict[str, Any]:
    """Return parameter-level density information for a dense PyTorch model."""
    total, nonzero = count_nonzero_weights(model)
    return {
        "parameter_count": int(count_parameters(model)),
        "total_weight_elements": int(total),
        "nonzero_weight_elements": int(nonzero),
        "sparsity": float(1.0 - (nonzero / total if total else 0.0)),
    }
