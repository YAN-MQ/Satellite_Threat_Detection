"""Compensation utilities for missing inter-plane model exchanges."""

from __future__ import annotations

from .aggregators import StateDict, scale_state_dict, add_state_dict


def compensate_missing_model(
    last_model: dict[str, object],
    current_plane_model: dict[str, object],
    rho: float,
) -> StateDict:
    """Build a compensation model when an inter-plane model is missing."""
    return add_state_dict(
        scale_state_dict(last_model, rho),
        scale_state_dict(current_plane_model, 1.0 - rho),
    )
