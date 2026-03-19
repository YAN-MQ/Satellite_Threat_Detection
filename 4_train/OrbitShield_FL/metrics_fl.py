"""Evaluation and metric helpers for federated experiments."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from .aggregators import StateDict


def make_eval_loader(features: np.ndarray, labels: np.ndarray, batch_size: int) -> DataLoader:
    """Create an evaluation dataloader from NumPy arrays."""
    dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def evaluate_global_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a model on a global validation or test split."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * features.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    sample_count = max(1, len(all_labels))
    return {
        "loss": total_loss / sample_count,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "label_distribution": dict(Counter(all_labels)),
    }


def summarize_round_metrics(
    round_idx: int,
    val_metrics: dict[str, object],
    test_metrics: dict[str, object] | None,
    communication_cost_mb: float,
    stale_update_ratio: float,
    link_failure_robustness: float,
) -> dict[str, object]:
    """Build a flat round metric record for CSV/JSON export."""
    test_accuracy = float(test_metrics["accuracy"]) if test_metrics is not None else float("nan")
    test_precision = float(test_metrics["precision"]) if test_metrics is not None else float("nan")
    test_recall = float(test_metrics["recall"]) if test_metrics is not None else float("nan")
    test_f1 = float(test_metrics["f1"]) if test_metrics is not None else float("nan")
    confusion = test_metrics["confusion_matrix"] if test_metrics is not None else []
    return {
        "round": round_idx,
        "val_loss": round(float(val_metrics["loss"]), 6),
        "val_accuracy": round(float(val_metrics["accuracy"]), 6),
        "val_precision": round(float(val_metrics["precision"]), 6),
        "val_recall": round(float(val_metrics["recall"]), 6),
        "val_f1": round(float(val_metrics["f1"]), 6),
        "test_accuracy": round(test_accuracy, 6),
        "test_precision": round(test_precision, 6),
        "test_recall": round(test_recall, 6),
        "test_f1": round(test_f1, 6),
        "communication_cost_mb": round(float(communication_cost_mb), 6),
        "stale_update_ratio": round(float(stale_update_ratio), 6),
        "link_failure_robustness": round(float(link_failure_robustness), 6),
        "confusion_matrix": str(confusion),
    }
