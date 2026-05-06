#!/usr/bin/env python3
"""Run the Level 4B ns-3 + libtorch federated runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset_profiles import get_dataset_profile


PROJECT_ROOT = Path("/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation")
BUILD_ROOT = PROJECT_ROOT.parent.parent
TORCH_LIB_DIR = Path("/home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/lib")
TORCH_CMAKE_PREFIX = Path("/home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/share/cmake")
TORCH_CMAKE_DIR = TORCH_CMAKE_PREFIX / "Torch"
CAFFE2_CMAKE_DIR = TORCH_CMAKE_PREFIX / "Caffe2"
VENV_PYTHON = "/home/lithic/final/ns3-gpu-venv/bin/python"
TRAIN_ROOT = PROJECT_ROOT / "4_train"

STATE_KEY_REMAP = {
    "cbam.channel_attention.fc.0.weight": "cbam.channel.fc1.weight",
    "cbam.channel_attention.fc.2.weight": "cbam.channel.fc2.weight",
    "cbam.spatial_attention.conv.weight": "cbam.spatial.conv.weight",
    "fc.0.weight": "fc1.weight",
    "fc.0.bias": "fc1.bias",
    "fc.3.weight": "fc2.weight",
    "fc.3.bias": "fc2.bias",
}

DTYPE_TO_NAME = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OrbitShield_FL Level 4B with ns-3 + libtorch")
    parser.add_argument("--dataset", choices=["cicids17", "sti"], default="cicids17")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--export_dir", default=None)
    parser.add_argument("--num_clients", type=int, default=12)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--partition_mode", choices=["iid", "dirichlet", "quantity_skew", "hybrid"], default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--force_export_init_state", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--conv_dim", type=int, default=32)
    parser.add_argument("--dsc_dim", type=int, default=64)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _resolve_train_root_path(path_like: str | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = TRAIN_ROOT / path
    return path.resolve()


def ensure_export(args: argparse.Namespace, profile) -> Path:
    export_dir = _resolve_train_root_path(args.export_dir) or (
        PROJECT_ROOT / "4_train" / "libtorch_data" / profile.name
    ).resolve()
    metadata = export_dir / "metadata.json"
    if metadata.exists() and not args.force_export:
        return export_dir

    run(
        [
            VENV_PYTHON,
            str(PROJECT_ROOT / "4_train" / "scripts" / "export_libtorch_dataset.py"),
            "--dataset",
            profile.name,
            "--output_dir",
            str(export_dir),
            "--num_clients",
            str(args.num_clients),
            "--partition_mode",
            args.partition_mode,
            "--dirichlet_alpha",
            str(args.dirichlet_alpha),
            "--seed",
            str(args.seed),
        ],
        cwd=PROJECT_ROOT / "4_train",
    )
    return export_dir


def _resolve_checkpoint_path(checkpoint: str | None) -> Path | None:
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.is_absolute():
        path = TRAIN_ROOT / path
    return path.resolve()


def _save_raw_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().contiguous().numpy()
    np.asarray(array).tofile(path)


def ensure_init_state(args: argparse.Namespace, profile) -> Path | None:
    if args.from_scratch:
        return None
    checkpoint_path = _resolve_checkpoint_path(args.init_checkpoint or profile.init_checkpoint)
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    state_dir = TRAIN_ROOT / "libtorch_init" / profile.name
    metadata_path = state_dir / "metadata.json"
    if metadata_path.exists() and not args.force_export_init_state:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_checkpoint") == str(checkpoint_path):
            return state_dir

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint structure for warm start: {checkpoint_path}")

    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    exported_keys: list[str] = []
    for index, (key, value) in enumerate(state.items()):
        tensor = value.detach().cpu().contiguous()
        target_key = STATE_KEY_REMAP.get(key, key)
        dtype_name = DTYPE_TO_NAME.get(tensor.dtype)
        if dtype_name is None:
            raise TypeError(f"Unsupported tensor dtype for warm start export: {key} -> {tensor.dtype}")
        safe_name = target_key.replace(".", "__")
        file_name = f"{index:03d}_{safe_name}.bin"
        _save_raw_tensor(state_dir / file_name, tensor)
        shape_spec = ",".join(str(dim) for dim in tensor.shape)
        manifest_lines.append(f"{target_key}\t{dtype_name}\t{shape_spec}\t{file_name}")
        exported_keys.append(target_key)

    (state_dir / "state_manifest.tsv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    metadata = {
        "dataset": profile.name,
        "source_checkpoint": str(checkpoint_path),
        "tensor_count": len(exported_keys),
        "exported_keys": exported_keys,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return state_dir


def maybe_build() -> None:
    run(
        [
            "cmake",
            "-S",
            str(BUILD_ROOT),
            "-B",
            str(BUILD_ROOT / "build"),
            "-DORBITSHIELD_ENABLE_LIBTORCH_RUNTIME=ON",
            f"-DCMAKE_PREFIX_PATH={TORCH_CMAKE_PREFIX}",
            f"-DTorch_DIR={TORCH_CMAKE_DIR}",
            f"-DCaffe2_DIR={CAFFE2_CMAKE_DIR}",
        ],
        cwd=BUILD_ROOT,
    )
    run(
        [
            "cmake",
            "--build",
            str(BUILD_ROOT / "build"),
            "--target",
            "scratch_06_realtime_emulation_federated_libtorch_runtime",
            "-j2",
        ],
        cwd=BUILD_ROOT,
    )


def main() -> None:
    args = parse_args()
    profile = get_dataset_profile(args.dataset)
    export_dir = ensure_export(args, profile)
    init_state_dir = ensure_init_state(args, profile)
    metadata = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    if args.build:
        maybe_build()

    binary = BUILD_ROOT / "build" / "scratch" / "06_realtime_emulation" / "ns3.46.1-federated_libtorch_runtime-optimized"
    if not binary.exists():
        raise FileNotFoundError(f"Missing 4B runtime binary: {binary}")

    output_dir = Path(
        _resolve_train_root_path(args.output_dir)
        or (PROJECT_ROOT / "4_train" / "experiments" / "OrbitShield_FL_ns3_libtorch" / profile.name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "LD_LIBRARY_PATH": f"{TORCH_LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
    }
    run(
        [
            str(binary),
            f"--data-dir={export_dir}",
            f"--output-dir={output_dir}",
            f"--dataset={profile.name}",
            f"--train-samples={metadata['train_shape'][0]}",
            f"--val-samples={metadata['val_shape'][0]}",
            f"--test-samples={metadata['test_shape'][0]}",
            f"--num-clients={args.num_clients}",
            f"--num-planes={args.num_planes}",
            f"--rounds={args.rounds}",
            f"--local-epochs={args.local_epochs}",
            f"--batch-size={args.batch_size}",
            f"--input-dim={profile.input_dim}",
            f"--seq-len={profile.seq_len}",
            f"--num-classes={profile.num_classes}",
            f"--hidden-dim={args.hidden_dim}",
            f"--conv-dim={args.conv_dim}",
            f"--dsc-dim={args.dsc_dim}",
            f"--fc-hidden=64",
            f"--dropout={args.dropout}",
            f"--device={args.device}",
            *([f"--init-state-dir={init_state_dir}"] if init_state_dir is not None else []),
        ],
        cwd=PROJECT_ROOT,
        extra_env=env,
    )


if __name__ == "__main__":
    main()
