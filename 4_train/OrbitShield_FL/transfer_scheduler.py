"""Model-transfer scheduling helpers for ns-3-driven federated communication."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .aggregators import StateDict, state_dict_num_bytes
from .topology import LinkState


@dataclass(frozen=True)
class TransferPlan:
    """Transfer feasibility summary for one model exchange."""

    model_size_bytes: int
    effective_bandwidth_mbps: float
    transfer_time_seconds: float
    contact_window_seconds: float
    can_finish: bool


def model_state_size_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    """Return the serialized dense parameter size for a model state dict."""
    return state_dict_num_bytes(state_dict)


def estimate_effective_bandwidth_mbps(
    nominal_bandwidth_mbps: float,
    packet_loss: float,
    success: bool,
) -> float:
    """Estimate effective usable throughput from nominal bandwidth and packet loss."""
    if not success or nominal_bandwidth_mbps <= 0:
        return 0.0
    usable_fraction = max(0.0, 1.0 - float(packet_loss))
    return max(0.0, float(nominal_bandwidth_mbps) * usable_fraction)


def estimate_transfer_time_seconds(
    model_size_bytes: int,
    effective_bandwidth_mbps: float,
) -> float:
    """Estimate the time needed to transfer a model under a given effective bandwidth."""
    if model_size_bytes <= 0:
        return 0.0
    if effective_bandwidth_mbps <= 0:
        return float("inf")
    bits = float(model_size_bytes) * 8.0
    bits_per_second = float(effective_bandwidth_mbps) * 1_000_000.0
    return bits / bits_per_second


def build_transfer_plan(
    model_size_bytes: int,
    nominal_bandwidth_mbps: float,
    packet_loss: float,
    contact_window_seconds: float,
    success: bool,
) -> TransferPlan:
    """Build a transfer-feasibility record for one communication window."""
    effective_bandwidth = estimate_effective_bandwidth_mbps(
        nominal_bandwidth_mbps=nominal_bandwidth_mbps,
        packet_loss=packet_loss,
        success=success,
    )
    transfer_time = estimate_transfer_time_seconds(
        model_size_bytes=model_size_bytes,
        effective_bandwidth_mbps=effective_bandwidth,
    )
    contact_window = max(0.0, float(contact_window_seconds))
    can_finish = success and contact_window > 0.0 and transfer_time <= contact_window
    return TransferPlan(
        model_size_bytes=int(model_size_bytes),
        effective_bandwidth_mbps=float(effective_bandwidth),
        transfer_time_seconds=float(transfer_time),
        contact_window_seconds=float(contact_window),
        can_finish=bool(can_finish),
    )


def build_transfer_plan_from_link(
    model_size_bytes: int,
    link_state: LinkState,
    bandwidth_mbps: float,
) -> TransferPlan:
    """Build a transfer plan from a LinkState plus externally supplied bandwidth."""
    return build_transfer_plan(
        model_size_bytes=model_size_bytes,
        nominal_bandwidth_mbps=bandwidth_mbps,
        packet_loss=link_state.packet_loss,
        contact_window_seconds=float(link_state.contact_duration),
        success=link_state.available and link_state.success,
    )


def can_transfer_model(
    model_size_bytes: int,
    link_state: LinkState,
    bandwidth_mbps: float,
) -> bool:
    """Return whether a model can finish transfer inside the current link window."""
    return build_transfer_plan_from_link(
        model_size_bytes=model_size_bytes,
        link_state=link_state,
        bandwidth_mbps=bandwidth_mbps,
    ).can_finish
