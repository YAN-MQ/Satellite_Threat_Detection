"""Inter-plane asynchronous gossip operations."""

from __future__ import annotations

import math
import numpy as np

from .aggregators import StateDict, weighted_average_state_dict
from .compensation import compensate_missing_model


def inter_plane_gossip(
    plane_id: int,
    self_model: StateDict,
    plane_neighbors: list[int],
    available_models: dict[int, StateDict],
    available_scores: dict[int, float],
    plane_model_cache: dict[int, StateDict],
    plane_staleness: dict[int, int],
    beta: float,
    beta_floor: float,
    lambda_s: float,
    method: str,
    rho: float,
) -> tuple[StateDict, dict[str, object]]:
    """Mix a plane model with neighbor plane models using asynchronous gossip."""
    if method not in {"intra_gossip", "full"} or not plane_neighbors:
        return self_model, {"used_neighbors": [], "missing_neighbors": [], "adaptive_beta": 0.0}

    neighbor_models = []
    neighbor_scores = []
    missing_neighbors = []

    for neighbor in plane_neighbors:
        if neighbor in available_models:
            neighbor_models.append(available_models[neighbor])
            quality = available_scores.get(neighbor, 1.0)
            staleness = plane_staleness.get(neighbor, 0)
            neighbor_scores.append(float(quality * math.exp(-lambda_s * staleness)))
        else:
            missing_neighbors.append(neighbor)
            if method == "full" and neighbor in plane_model_cache:
                neighbor_models.append(compensate_missing_model(plane_model_cache[neighbor], self_model, rho))
                staleness = plane_staleness.get(neighbor, 1)
                neighbor_scores.append(float(0.5 * math.exp(-lambda_s * staleness)))

    if not neighbor_models:
        return self_model, {"used_neighbors": [], "missing_neighbors": missing_neighbors, "adaptive_beta": 0.0}

    gamma = np.asarray(neighbor_scores, dtype=np.float64)
    if gamma.sum() <= 0:
        gamma = np.full(len(neighbor_models), 1.0 / len(neighbor_models))
    else:
        gamma = gamma / gamma.sum()
    mixed_neighbors = weighted_average_state_dict(neighbor_models, gamma)
    avg_neighbor_quality = float(np.mean(neighbor_scores)) if neighbor_scores else 0.0
    adaptive_beta = beta_floor + (beta - beta_floor) * max(0.0, min(1.0, avg_neighbor_quality))
    updated = weighted_average_state_dict([self_model, mixed_neighbors], [1.0 - adaptive_beta, adaptive_beta])
    used_neighbors = [neighbor for neighbor in plane_neighbors if neighbor not in missing_neighbors]
    return updated, {
        "used_neighbors": used_neighbors,
        "missing_neighbors": missing_neighbors,
        "adaptive_beta": float(adaptive_beta),
        "gamma": {str(neighbor): float(weight) for neighbor, weight in zip(used_neighbors + missing_neighbors, gamma)},
    }
