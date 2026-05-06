"""Aggregation helpers for robust federated satellite learning."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch


StateDict = OrderedDict[str, torch.Tensor]


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> StateDict:
    """Clone a model state dict onto CPU tensors."""
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in state_dict.items())


def subtract_state_dict(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> StateDict:
    """Compute the tensor-wise difference a - b."""
    return OrderedDict((key, a[key].detach().cpu() - b[key].detach().cpu()) for key in a)


def add_state_dict(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> StateDict:
    """Compute the tensor-wise sum a + b."""
    return OrderedDict((key, a[key].detach().cpu() + b[key].detach().cpu()) for key in a)


def scale_state_dict(state_dict: dict[str, torch.Tensor], factor: float) -> StateDict:
    """Scale a state dict by a scalar factor."""
    return OrderedDict((key, value.detach().cpu() * factor) for key, value in state_dict.items())


def weighted_average_state_dict(
    models: Iterable[dict[str, torch.Tensor]],
    weights: Iterable[float],
) -> StateDict:
    """Compute a weighted average of model state dicts."""
    models = list(models)
    weights = np.asarray(list(weights), dtype=np.float64)
    if not models:
        raise ValueError("weighted_average_state_dict requires at least one model")
    if weights.sum() <= 0:
        weights = np.full(len(models), 1.0 / len(models))
    else:
        weights = weights / weights.sum()

    averaged = OrderedDict()
    for key in models[0]:
        averaged[key] = sum(model[key].detach().cpu() * float(weight) for model, weight in zip(models, weights))
    return averaged


def flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single vector."""
    return torch.cat([value.detach().cpu().reshape(-1) for value in state_dict.values()])


def cosine_similarity_state_dict(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """Compute cosine similarity between two model updates."""
    vec_a = flatten_state_dict(a)
    vec_b = flatten_state_dict(b)
    denom = torch.norm(vec_a) * torch.norm(vec_b)
    if torch.isclose(denom, torch.tensor(0.0)):
        return 0.0
    return float(torch.dot(vec_a, vec_b) / denom)


def state_dict_num_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    """Estimate communication payload size for a dense model update."""
    total = 0
    for value in state_dict.values():
        total += value.numel() * value.element_size()
    return int(total)


def compute_staleness(last_sync_round: int, current_round: int) -> int:
    """Return the round staleness of a client or plane model."""
    return max(0, current_round - last_sync_round)


def estimate_link_quality(
    success_rate: float,
    contact_duration: int,
    delay: float,
    packet_loss: float,
) -> float:
    """Estimate a bounded [0, 1] link-quality score."""
    duration_term = min(1.0, contact_duration / max(contact_duration, 1))
    delay_term = math.exp(-delay / 100.0)
    loss_term = max(0.0, 1.0 - packet_loss)
    score = success_rate * 0.45 + duration_term * 0.15 + delay_term * 0.20 + loss_term * 0.20
    return float(np.clip(score, 0.0, 1.0))


def intra_plane_aggregate(
    client_payloads: list[dict[str, object]],
    current_round: int,
    lambda_s: float,
    method: str,
) -> tuple[StateDict, dict[str, object]]:
    """Aggregate satellite models within the same orbital plane."""
    if not client_payloads:
        raise ValueError("intra_plane_aggregate requires non-empty client payloads")

    if method in {"fedavg", "intra_only", "intra_gossip"}:
        scores = np.asarray([float(payload["sample_count"]) for payload in client_payloads], dtype=np.float64)
    else:
        scores = []
        for payload in client_payloads:
            staleness = compute_staleness(int(payload["last_sync_round"]), current_round)
            score = (
                float(payload["sample_count"])
                * math.exp(-lambda_s * staleness)
                * float(payload["link_quality"])
                * float(payload["reputation"])
            )
            scores.append(score)
        scores = np.asarray(scores, dtype=np.float64)

    weights = scores / scores.sum() if scores.sum() > 0 else np.full(len(client_payloads), 1.0 / len(client_payloads))
    aggregated_model = weighted_average_state_dict(
        [payload["weights"] for payload in client_payloads],
        weights,
    )
    metadata = {
        "client_weights": {
            str(payload["client_id"]): float(weight) for payload, weight in zip(client_payloads, weights)
        },
        "stale_clients": int(
            sum(compute_staleness(int(payload["last_sync_round"]), current_round) > 0 for payload in client_payloads)
        ),
        "participant_count": len(client_payloads),
    }
    return aggregated_model, metadata
