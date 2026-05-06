"""Adapters that convert ns-3 constellation traces into federated topology snapshots."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .ns3_bridge import Ns3RoundTrace, Ns3TraceBundle, load_ns3_round_trace, load_ns3_trace_bundle
from .topology import LinkState


def _to_bool(value: Any) -> bool:
    """Convert a JSON field to a boolean."""
    return bool(value)


def _to_float(value: Any) -> float:
    """Convert a JSON field to a float."""
    return float(value)


def _to_int(value: Any) -> int:
    """Convert a JSON field to an integer."""
    return int(value)


def _link_state_from_payload(payload: dict[str, object]) -> LinkState:
    """Convert a serialized ns-3 link record into the internal LinkState format."""
    return LinkState(
        available=_to_bool(payload["available"]),
        success=_to_bool(payload["success"]),
        delay=_to_float(payload["delay_ms"]),
        packet_loss=_to_float(payload["packet_loss"]),
        contact_duration=_to_int(round(_to_float(payload["contact_duration_s"]))),
    )


def convert_ns3_round_trace(round_trace: Ns3RoundTrace) -> dict[str, dict[tuple[int, int], LinkState] | dict[int, list[int]]]:
    """Convert one ns-3 round trace into the federated topology snapshot schema."""
    payload = round_trace.payload

    num_planes = int(payload["num_planes"])
    plane_neighbors: dict[int, list[int]] = {plane_id: [] for plane_id in range(num_planes)}
    intra_plane_links: dict[tuple[int, int], LinkState] = {}
    inter_plane_links: dict[tuple[int, int], LinkState] = {}
    intra_plane_bandwidth_mbps: dict[int, float] = {}
    inter_plane_bandwidth_mbps: dict[tuple[int, int], float] = {}

    serialized_intra = payload["intra_plane_links"]
    if not isinstance(serialized_intra, dict):
        raise ValueError(f"intra_plane_links must be a dict in {round_trace.path}")
    for plane_key, state_payload in serialized_intra.items():
        plane_id = int(plane_key)
        if not isinstance(state_payload, dict):
            raise ValueError(f"Invalid intra-plane link payload for plane {plane_key}")
        intra_plane_links[(plane_id, plane_id)] = _link_state_from_payload(state_payload)
        intra_plane_bandwidth_mbps[plane_id] = _to_float(state_payload["bandwidth_mbps"])

    serialized_inter = payload["inter_plane_links"]
    if not isinstance(serialized_inter, dict):
        raise ValueError(f"inter_plane_links must be a dict in {round_trace.path}")
    for pair_key, state_payload in serialized_inter.items():
        if not isinstance(state_payload, dict):
            raise ValueError(f"Invalid inter-plane link payload for pair {pair_key}")
        plane_a_str, plane_b_str = str(pair_key).split("-")
        plane_a = int(plane_a_str)
        plane_b = int(plane_b_str)
        pair = (min(plane_a, plane_b), max(plane_a, plane_b))
        state = _link_state_from_payload(state_payload)
        inter_plane_links[pair] = state
        inter_plane_bandwidth_mbps[pair] = _to_float(state_payload["bandwidth_mbps"])
        if state.available:
            plane_neighbors[pair[0]].append(pair[1])
            plane_neighbors[pair[1]].append(pair[0])

    for plane_id in plane_neighbors:
        plane_neighbors[plane_id] = sorted(set(plane_neighbors[plane_id]))

    return {
        "plane_neighbors": plane_neighbors,
        "inter_plane_links": inter_plane_links,
        "intra_plane_links": intra_plane_links,
        "intra_plane_bandwidth_mbps": intra_plane_bandwidth_mbps,
        "inter_plane_bandwidth_mbps": inter_plane_bandwidth_mbps,
    }


def load_ns3_topology_snapshot(trace_dir: str, round_idx: int) -> dict[str, dict[tuple[int, int], LinkState] | dict[int, list[int]]]:
    """Load and convert one ns-3 round trace into a federated topology snapshot."""
    round_trace = load_ns3_round_trace(trace_dir, round_idx)
    return convert_ns3_round_trace(round_trace)


def load_all_ns3_topology_snapshots(
    trace_dir: str,
) -> tuple[Ns3TraceBundle, dict[int, dict[str, dict[tuple[int, int], LinkState] | dict[int, list[int]]]]]:
    """Load a full ns-3 trace directory and convert all round traces."""
    bundle = load_ns3_trace_bundle(trace_dir)
    snapshots = {round_trace.round_idx: convert_ns3_round_trace(round_trace) for round_trace in bundle.rounds}
    return bundle, snapshots


def serialize_federated_topology(
    topology: dict[str, dict[tuple[int, int], LinkState] | dict[int, list[int]]],
) -> dict[str, object]:
    """Serialize a converted topology snapshot for debugging or artifact export."""
    inter_plane_links = {
        f"{pair[0]}-{pair[1]}": asdict(state) for pair, state in topology["inter_plane_links"].items()
    }
    intra_plane_links = {
        f"{pair[0]}-{pair[1]}": asdict(state) for pair, state in topology["intra_plane_links"].items()
    }
    return {
        "plane_neighbors": topology["plane_neighbors"],
        "inter_plane_links": inter_plane_links,
        "intra_plane_links": intra_plane_links,
        "intra_plane_bandwidth_mbps": {
            str(plane_id): float(bandwidth)
            for plane_id, bandwidth in topology.get("intra_plane_bandwidth_mbps", {}).items()
        },
        "inter_plane_bandwidth_mbps": {
            f"{pair[0]}-{pair[1]}": float(bandwidth)
            for pair, bandwidth in topology.get("inter_plane_bandwidth_mbps", {}).items()
        },
    }
