"""Lightweight dynamic topology simulator for a multi-plane constellation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinkState:
    """State for a simulated communication link."""

    available: bool
    success: bool
    delay: float
    packet_loss: float
    contact_duration: int


def build_plane_assignments(num_clients: int, num_planes: int) -> dict[str, int]:
    """Map each satellite client to a plane id."""
    sats_per_plane = num_clients // num_planes
    return {f"sat_{idx}": idx // sats_per_plane for idx in range(num_clients)}


def generate_topology_snapshot(
    current_round: int,
    num_planes: int,
    intra_plane_success_prob: float,
    inter_plane_success_prob: float,
    inter_plane_contact_period: int,
    inter_plane_contact_duration: int,
    packet_loss_prob: float,
    link_delay_mean: float,
    seed: int,
) -> dict[str, dict[tuple[int, int], LinkState] | dict[int, list[int]]]:
    """Generate the intra-plane and inter-plane link state for a round."""
    rng = np.random.default_rng(seed + current_round)

    plane_neighbors: dict[int, list[int]] = {plane_id: [] for plane_id in range(num_planes)}
    inter_plane_links: dict[tuple[int, int], LinkState] = {}

    for plane_id in range(num_planes):
        next_plane = (plane_id + 1) % num_planes
        pair = tuple(sorted((plane_id, next_plane)))
        phase_offset = pair[0]
        visible = ((current_round + phase_offset) % inter_plane_contact_period) < inter_plane_contact_duration
        success = bool(visible and rng.random() < inter_plane_success_prob)
        link_state = LinkState(
            available=visible,
            success=success,
            delay=float(rng.exponential(link_delay_mean)),
            packet_loss=float(packet_loss_prob if visible else 1.0),
            contact_duration=inter_plane_contact_duration if visible else 0,
        )
        inter_plane_links[pair] = link_state
        if visible:
            plane_neighbors[plane_id].append(next_plane)
            plane_neighbors[next_plane].append(plane_id)

    intra_plane_links: dict[tuple[int, int], LinkState] = {}
    for plane_id in range(num_planes):
        intra_plane_links[(plane_id, plane_id)] = LinkState(
            available=True,
            success=bool(rng.random() < intra_plane_success_prob),
            delay=float(rng.exponential(link_delay_mean / 2.0)),
            packet_loss=float(packet_loss_prob / 2.0),
            contact_duration=inter_plane_contact_duration,
        )

    return {
        "plane_neighbors": plane_neighbors,
        "inter_plane_links": inter_plane_links,
        "intra_plane_links": intra_plane_links,
    }
