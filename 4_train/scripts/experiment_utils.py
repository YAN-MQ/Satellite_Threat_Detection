#!/usr/bin/env python3
"""Shared helpers for ablation, comparison, and visualization scripts."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data import create_dataloaders, load_npz_data
from src.training import Trainer, get_optimizer, get_scheduler


DEFAULT_CLASS_NAMES = ["Benign", "DDoS", "PortScan"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_window_data(data_dir: str):
    return load_npz_data(data_dir)


def _stratified_subset(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return x, y

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    target = np.maximum(1, np.floor(proportions * max_samples).astype(int))
    target = np.minimum(target, counts)

    while target.sum() < max_samples:
        deficits = counts - target
        candidates = np.where(deficits > 0)[0]
        if len(candidates) == 0:
            break
        best = candidates[np.argmax(proportions[candidates] - (target[candidates] / max_samples))]
        target[best] += 1

    while target.sum() > max_samples:
        candidates = np.where(target > 1)[0]
        if len(candidates) == 0:
            break
        worst = candidates[np.argmax(target[candidates] / max_samples - proportions[candidates])]
        target[worst] -= 1

    selected = []
    for cls, take in zip(classes, target):
        cls_indices = np.flatnonzero(y == cls)
        if take >= len(cls_indices):
            chosen = cls_indices
        else:
            chosen = rng.choice(cls_indices, size=int(take), replace=False)
        selected.append(np.sort(chosen))

    indices = np.sort(np.concatenate(selected))
    return x[indices], y[indices]


def make_window_loaders(
    data_dir: str,
    batch_size: int = 64,
    max_samples: int | None = None,
    seed: int = 42,
):
    x_train, y_train, x_val, y_val, x_test, y_test = load_window_data(data_dir)
    if max_samples is not None:
        x_train, y_train = _stratified_subset(x_train, y_train, max_samples, seed=seed)
        x_val, y_val = _stratified_subset(x_val, y_val, max_samples, seed=seed + 1)
        x_test, y_test = _stratified_subset(x_test, y_test, max_samples, seed=seed + 2)
    loaders = create_dataloaders(
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
    return (x_train, y_train, x_val, y_val, x_test, y_test), loaders


def benchmark_inference_latency(
    model: nn.Module,
    sample_shape: tuple[int, int, int],
    device: torch.device,
    warmup_iters: int = 20,
    measure_iters: int = 100,
) -> float:
    """Measure pure forward latency per batch in seconds."""
    model.eval()
    dummy = torch.randn(*sample_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(measure_iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    return elapsed / measure_iters


def train_deep_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_path: str,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name="adamw", lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_name="plateau")
    trainer = Trainer(model, criterion, optimizer, scheduler, device)

    start_time = time.time()
    history = trainer.train(train_loader, val_loader, num_epochs=epochs, save_path=save_path)
    train_time = time.time() - start_time

    metrics = trainer.evaluate(test_loader)
    first_batch = next(iter(test_loader))[0]
    inference_time = benchmark_inference_latency(
        model,
        sample_shape=tuple(first_batch.shape),
        device=device,
    )

    return {
        "history": history,
        "metrics": metrics,
        "training_time_sec": train_time,
        "inference_time_sec": inference_time,
    }


def confusion_matrix_to_list(metrics: dict[str, Any]) -> dict[str, Any]:
    converted = dict(metrics)
    if "confusion_matrix" in converted and hasattr(converted["confusion_matrix"], "tolist"):
        converted["confusion_matrix"] = converted["confusion_matrix"].tolist()
    return converted


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(
    name: str,
    family: str,
    params: int | str,
    flops: int | str,
    result: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "model": name,
        "family": family,
        "params": params,
        "flops": flops,
        "accuracy": round(float(metrics["accuracy"]), 6),
        "precision": round(float(metrics["precision"]), 6),
        "recall": round(float(metrics["recall"]), 6),
        "f1": round(float(metrics["f1"]), 6),
        "training_time_sec": round(float(result["training_time_sec"]), 4),
        "inference_time_sec": round(float(result["inference_time_sec"]), 4),
        "confusion_matrix": json.dumps(confusion_matrix_to_list(metrics)["confusion_matrix"]),
    }


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.lower() != "n/a":
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalize_higher(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _normalize_lower(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(high - value) / (high - low) for value in values]


def add_metric_ranks(rows: list[dict[str, Any]], metrics: list[str] | None = None) -> list[dict[str, Any]]:
    metrics = metrics or ["accuracy", "f1"]
    deep_rows = []
    for row in rows:
        if all(_safe_float(row.get(metric)) is not None for metric in metrics):
            deep_rows.append(row)

    for metric in metrics:
        ranked = sorted(deep_rows, key=lambda row: float(row[metric]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row[f"{metric}_rank"] = rank

    for row in rows:
        for metric in metrics:
            row.setdefault(f"{metric}_rank", "n/a")

    return rows


def add_composite_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deep_rows = []
    deep_indices = []
    for idx, row in enumerate(rows):
        params = _safe_float(row.get("params"))
        flops = _safe_float(row.get("flops"))
        latency = _safe_float(row.get("inference_time_sec"))
        if params is None or flops is None or latency is None:
            row["composite_score"] = "n/a"
            row["composite_rank"] = "n/a"
            continue
        deep_rows.append(row)
        deep_indices.append(idx)

    if not deep_rows:
        return rows

    acc_norm = _normalize_higher([float(row["accuracy"]) for row in deep_rows])
    f1_norm = _normalize_higher([float(row["f1"]) for row in deep_rows])
    params_norm = _normalize_lower([float(row["params"]) for row in deep_rows])
    flops_norm = _normalize_lower([float(row["flops"]) for row in deep_rows])
    latency_norm = _normalize_lower([float(row["inference_time_sec"]) for row in deep_rows])

    scores = []
    for row, acc, f1, params, flops, latency in zip(
        deep_rows, acc_norm, f1_norm, params_norm, flops_norm, latency_norm
    ):
        score = (
            0.30 * acc
            + 0.20 * f1
            + 0.25 * params
            + 0.15 * flops
            + 0.10 * latency
        )
        row["composite_score"] = round(score, 6)
        scores.append(score)

    ranked = sorted(
        ((idx, rows[idx]["composite_score"]) for idx in deep_indices),
        key=lambda item: item[1],
        reverse=True,
    )
    for rank, (idx, _) in enumerate(ranked, start=1):
        rows[idx]["composite_rank"] = rank

    return rows
