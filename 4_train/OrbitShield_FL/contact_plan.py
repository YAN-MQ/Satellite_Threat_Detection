"""Periodic contact plan helpers for inter-plane visibility."""

from __future__ import annotations


def is_contact_active(
    current_round: int,
    plane_a: int,
    plane_b: int,
    contact_period: int,
    contact_duration: int,
) -> bool:
    """Return whether two planes are in a contact window for the current round."""
    phase_offset = min(plane_a, plane_b)
    return ((current_round + phase_offset) % contact_period) < contact_duration
