#!/usr/bin/env python3
"""Export NPZ datasets and OrbitShield_FL partitions to libtorch-readable files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset_profiles import get_dataset_profile
from OrbitShield_FL.partition import dump_partition_stats, load_window_dataset, partition_train_dataset_for_satellites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CICIDS17/STI datasets for libtorch runtime")
    parser.add_argument("--dataset", choices=["cicids17", "sti"], default="cicids17")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_clients", type=int, default=12)
    parser.add_argument("--partition_mode", choices=["iid", "dirichlet", "quantity_skew", "hybrid"], default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.contiguous(), path)


def save_raw_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def main() -> None:
    args = parse_args()
    profile = get_dataset_profile(args.dataset)
    data_dir = Path(args.data_dir or profile.data_dir).resolve()
    output_dir = Path(
        args.output_dir
        or (Path(__file__).resolve().parent.parent / "libtorch_data" / profile.name)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_window_dataset(str(data_dir))
    train_x_np, train_y_np = dataset["train"]
    val_x_np, val_y_np = dataset["val"]
    test_x_np, test_y_np = dataset["test"]

    train_x = torch.from_numpy(train_x_np).float()
    train_y = torch.from_numpy(train_y_np).long()
    val_x = torch.from_numpy(val_x_np).float()
    val_y = torch.from_numpy(val_y_np).long()
    test_x = torch.from_numpy(test_x_np).float()
    test_y = torch.from_numpy(test_y_np).long()

    save_tensor(output_dir / "train_X.pt", train_x)
    save_tensor(output_dir / "train_y.pt", train_y)
    save_tensor(output_dir / "val_X.pt", val_x)
    save_tensor(output_dir / "val_y.pt", val_y)
    save_tensor(output_dir / "test_X.pt", test_x)
    save_tensor(output_dir / "test_y.pt", test_y)
    save_raw_array(output_dir / "train_X.f32", train_x_np.astype(np.float32, copy=False))
    save_raw_array(output_dir / "train_y.i64", train_y_np.astype(np.int64, copy=False))
    save_raw_array(output_dir / "val_X.f32", val_x_np.astype(np.float32, copy=False))
    save_raw_array(output_dir / "val_y.i64", val_y_np.astype(np.int64, copy=False))
    save_raw_array(output_dir / "test_X.f32", test_x_np.astype(np.float32, copy=False))
    save_raw_array(output_dir / "test_y.i64", test_y_np.astype(np.int64, copy=False))

    partition_map = partition_train_dataset_for_satellites(
        train_npz_path=str(data_dir / "train.npz"),
        num_clients=args.num_clients,
        mode=args.partition_mode,
        alpha=args.dirichlet_alpha,
        seed=args.seed,
    )
    partition_dir = output_dir / "partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)
    for client_id, info in partition_map.items():
        indices = torch.from_numpy(info["indices"]).long()
        save_tensor(partition_dir / f"{client_id}.pt", indices)
        save_raw_array(partition_dir / f"{client_id}.i64", np.asarray(info["indices"], dtype=np.int64))

    dump_partition_stats(str(output_dir / "partition_stats.json"), partition_map)

    metadata = {
        "dataset": profile.name,
        "data_dir": str(data_dir),
        "num_clients": args.num_clients,
        "partition_mode": args.partition_mode,
        "dirichlet_alpha": args.dirichlet_alpha,
        "seed": args.seed,
        "input_dim": profile.input_dim,
        "num_classes": profile.num_classes,
        "seq_len": profile.seq_len,
        "train_shape": list(train_x.shape),
        "val_shape": list(val_x.shape),
        "test_shape": list(test_x.shape),
        "class_names": profile.class_names,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Exported libtorch dataset to: {output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
