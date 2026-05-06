#!/usr/bin/env python3
"""Structured compression helpers and A1 smoke verification."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_DIR)

from src.models import StructuredDSC_CBAM_GRU, count_flops, count_parameters


FORMAL_CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints_gru_formal_tuned",
    "cicids17_gru_best.pt",
)

SOURCE_CFG = {
    "input_dim": 18,
    "num_classes": 3,
    "conv_dim": 16,
    "dsc_dim": 48,
    "hidden_dim": 32,
    "fc_hidden": 64,
    "dropout": 0.4,
    "bidirectional": False,
}

REQUIRED_TRANSFER_KEYS = [
    "conv.weight",
    "conv.bias",
    "dsc.dw.weight",
    "dsc.dw.bias",
    "dsc.pw.weight",
    "dsc.pw.bias",
    "dsc.bn.weight",
    "dsc.bn.bias",
    "dsc.bn.running_mean",
    "dsc.bn.running_var",
    "dsc.bn.num_batches_tracked",
    "cbam.channel_attention.fc.0.weight",
    "cbam.channel_attention.fc.2.weight",
    "cbam.spatial_attention.conv.weight",
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.bias_ih_l0",
    "gru.bias_hh_l0",
    "fc.0.weight",
    "fc.0.bias",
    "fc.3.weight",
    "fc.3.bias",
]

CANDIDATES = {
    "A1": {
        "conv_dim": 16,
        "dsc_dim": 32,
        "hidden_dim": 24,
        "fc_hidden": 32,
    },
    "A2": {
        "conv_dim": 16,
        "dsc_dim": 32,
        "hidden_dim": 16,
        "fc_hidden": 32,
    },
    "A3": {
        "conv_dim": 16,
        "dsc_dim": 24,
        "hidden_dim": 24,
        "fc_hidden": 24,
    },
}


def _slice_tensor(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if source.ndim != len(target_shape):
        raise ValueError(f"Rank mismatch: source {tuple(source.shape)} vs target {tuple(target_shape)}")

    slices = tuple(slice(0, min(source.shape[idx], target_shape[idx])) for idx in range(source.ndim))
    sliced = source[slices].clone()
    if tuple(sliced.shape) != tuple(target_shape):
        raise ValueError(f"Unsafe slice: source {tuple(source.shape)} -> sliced {tuple(sliced.shape)} target {tuple(target_shape)}")
    return sliced


def _safe_copy_param(target_state: dict[str, torch.Tensor], source_state: dict[str, torch.Tensor], key: str) -> None:
    if key not in target_state or key not in source_state:
        return
    target_state[key] = _slice_tensor(source_state[key], target_state[key].shape)


def _transfer_cbam_channel_attention(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> None:
    first_key = "cbam.channel_attention.fc.0.weight"
    second_key = "cbam.channel_attention.fc.2.weight"
    if first_key not in source_state or second_key not in source_state:
        return

    if first_key in target_state:
        target_state[first_key] = _slice_tensor(source_state[first_key], target_state[first_key].shape)
    if second_key in target_state:
        target_state[second_key] = _slice_tensor(source_state[second_key], target_state[second_key].shape)


def _transfer_gru(source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]) -> None:
    for key in [
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.bias_ih_l0",
        "gru.bias_hh_l0",
    ]:
        if key not in source_state or key not in target_state:
            continue
        target_state[key] = _slice_tensor(source_state[key], target_state[key].shape)


def transfer_structured_weights(
    source_state_dict: dict[str, torch.Tensor],
    target_model: torch.nn.Module,
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
) -> tuple[torch.nn.Module, list[str]]:
    """Transfer structured weights by leading-dimension slicing.

    Returns the initialized target model plus the ordered list of keys that were
    explicitly transferred.  The configs are validated so we do not silently
    transfer between incompatible source checkpoints.
    """
    if source_cfg["input_dim"] != target_cfg["input_dim"]:
        raise ValueError("input_dim mismatch between source and target configs")
    if source_cfg["num_classes"] != target_cfg["num_classes"]:
        raise ValueError("num_classes mismatch between source and target configs")
    if source_cfg["bidirectional"] != target_cfg["bidirectional"]:
        raise ValueError("bidirectional mismatch: structured transfer only supports same GRU directionality")

    target_state = target_model.state_dict()
    transferred: list[str] = []

    for key in [
        "conv.weight",
        "conv.bias",
        "dsc.dw.weight",
        "dsc.dw.bias",
        "dsc.pw.weight",
        "dsc.pw.bias",
        "dsc.bn.weight",
        "dsc.bn.bias",
        "dsc.bn.running_mean",
        "dsc.bn.running_var",
        "cbam.spatial_attention.conv.weight",
        "fc.0.weight",
        "fc.0.bias",
        "fc.3.weight",
        "fc.3.bias",
    ]:
        if key in target_state and key in source_state_dict:
            target_state[key] = _slice_tensor(source_state_dict[key], target_state[key].shape)
            transferred.append(key)

    if "dsc.bn.num_batches_tracked" in source_state_dict and "dsc.bn.num_batches_tracked" in target_state:
        target_state["dsc.bn.num_batches_tracked"] = source_state_dict["dsc.bn.num_batches_tracked"].clone()
        transferred.append("dsc.bn.num_batches_tracked")

    if "cbam.channel_attention.fc.0.weight" in target_state and "cbam.channel_attention.fc.0.weight" in source_state_dict:
        target_state["cbam.channel_attention.fc.0.weight"] = _slice_tensor(
            source_state_dict["cbam.channel_attention.fc.0.weight"],
            target_state["cbam.channel_attention.fc.0.weight"].shape,
        )
        transferred.append("cbam.channel_attention.fc.0.weight")
    if "cbam.channel_attention.fc.2.weight" in target_state and "cbam.channel_attention.fc.2.weight" in source_state_dict:
        target_state["cbam.channel_attention.fc.2.weight"] = _slice_tensor(
            source_state_dict["cbam.channel_attention.fc.2.weight"],
            target_state["cbam.channel_attention.fc.2.weight"].shape,
        )
        transferred.append("cbam.channel_attention.fc.2.weight")

    for key in [
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.bias_ih_l0",
        "gru.bias_hh_l0",
    ]:
        if key in target_state and key in source_state_dict:
            target_state[key] = _slice_tensor(source_state_dict[key], target_state[key].shape)
            transferred.append(key)

    missing = [key for key in REQUIRED_TRANSFER_KEYS if key not in transferred]
    if missing:
        raise ValueError(f"Structured transfer skipped required keys: {missing}")

    target_model.load_state_dict(target_state)
    return target_model, transferred


def build_candidate_model(candidate_name: str) -> StructuredDSC_CBAM_GRU:
    candidate_cfg = CANDIDATES[candidate_name]
    return StructuredDSC_CBAM_GRU(
        input_dim=SOURCE_CFG["input_dim"],
        num_classes=SOURCE_CFG["num_classes"],
        dropout=SOURCE_CFG["dropout"],
        bidirectional=SOURCE_CFG["bidirectional"],
        **candidate_cfg,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured compression smoke test utilities")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES.keys()), default="A1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(FORMAL_CHECKPOINT, map_location="cpu")
    candidate_cfg = {
        **SOURCE_CFG,
        **CANDIDATES[args.candidate],
    }
    model = build_candidate_model(args.candidate)
    model, transferred = transfer_structured_weights(checkpoint, model, SOURCE_CFG, candidate_cfg)
    model.eval()

    sample = torch.randn(2, 10, SOURCE_CFG["input_dim"])
    with torch.no_grad():
        output = model(sample)

    print(f"Loaded checkpoint: {FORMAL_CHECKPOINT}")
    print(f"Candidate: {args.candidate}")
    print(f"Transferred keys ({len(transferred)}): {transferred}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Approx FLOPs: {count_flops(model, input_size=(1, 10, SOURCE_CFG['input_dim'])):,}")
    print(f"Forward pass success: output shape {list(output.shape)}")


if __name__ == "__main__":
    main()
