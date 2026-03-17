"""Reputation scoring for federated satellite clients."""

from __future__ import annotations

import math

import torch

from .aggregators import StateDict, cosine_similarity_state_dict


def _bounded_improvement(loss_before: float, loss_after: float) -> float:
    """Map loss improvement to a bounded [0, 1] score."""
    delta = loss_before - loss_after
    return float((math.tanh(delta) + 1.0) / 2.0)


def compute_score(
    client_update: StateDict,
    plane_average_update: StateDict,
    loss_before: float,
    loss_after: float,
    stable_i: float,
    weights: tuple[float, float, float],
) -> tuple[float, dict[str, float]]:
    """Compute the reputation score for a client update."""
    sim_i = (cosine_similarity_state_dict(client_update, plane_average_update) + 1.0) / 2.0
    improve_i = _bounded_improvement(loss_before, loss_after)
    stable_i = float(max(0.0, min(1.0, stable_i)))
    c1, c2, c3 = weights
    score = c1 * sim_i + c2 * improve_i + c3 * stable_i
    return float(score), {
        "sim_i": float(sim_i),
        "improve_i": float(improve_i),
        "stable_i": float(stable_i),
    }


def update_reputation(
    reputation: float,
    score: float,
    mu: float,
    r_min: float,
) -> float:
    """Update client reputation using exponential smoothing and clipping."""
    updated = mu * reputation + (1.0 - mu) * score
    return float(max(r_min, min(1.0, updated)))
