#!/usr/bin/env python3
"""Prepare the STI CSV dataset into NPZ splits compatible with this project."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STI_DIR = os.path.join(BASE_DIR, "STI_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_sti")

DATASET_FILES = [
    ("STI_Benign.rar", "STI_Benign.csv", "Benign", 0),
    ("STI_Signal_disruption.rar", "STI_Signal_disruption.csv", "Signal Disruption", 1),
    ("STI_UDP_flood.rar", "STI_UDP_flood.csv", "UDP flood", 2),
    ("STI_Jamming.rar", "STI_Jamming.csv", "Jamming", 3),
    ("STI_Bruteforce.rar", "STI_Bruteforce.csv", "Bruteforce", 4),
    ("STI_Infiltration.rar", "STI_Infiltration.csv", "Infiltration", 5),
    ("STI_DoS.rar", "STI_DoS.csv", "DoS", 6),
    ("STI_DDoS.rar", "STI_DDoS.csv", "DDoS", 7),
]

FEATURE_COLUMNS = [
    "Duration",
    "Size",
    "Protocol",
    "Sinr",
    "Throughput",
    "Flow_bytes_s",
    "Flow_packets_s",
    "Inv_mean",
    "Inv_min",
    "Inv_max",
    "DNS_query_id",
    "L7_protocol",
    "DNS_type",
    "TTL_min",
    "TTL_max",
    "DNS_TTL_answer",
    "Next_Current_diff",
    "Next_Pre_diff",
    "SNext_Current_diff",
    "SNext_Pre_diff",
]


@dataclass
class SplitBundle:
    """Container for train, validation, and test arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare the STI dataset for model training")
    parser.add_argument("--sti-dir", default=STI_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="Apply MinMax normalization again on the train split before exporting",
    )
    return parser.parse_args()


def require_bsdtar() -> str:
    """Return the path to bsdtar or raise an error."""
    path = shutil.which("bsdtar")
    if path is None:
        raise RuntimeError("bsdtar is required to read STI .rar files")
    return path


def read_csv_from_rar(bsdtar_path: str, rar_path: str, csv_name: str) -> pd.DataFrame:
    """Read a CSV file directly from a RAR archive."""
    completed = subprocess.run(
        [bsdtar_path, "-xOf", rar_path, csv_name],
        check=True,
        capture_output=True,
    )
    frame = pd.read_csv(io.BytesIO(completed.stdout))
    if frame.columns[0].startswith("Unnamed") or frame.columns[0] == "":
        frame = frame.drop(columns=frame.columns[0])
    return frame


def split_frame(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a class-specific dataframe in temporal order."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def encode_and_stack(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_id: int,
    renormalize: bool,
) -> SplitBundle:
    """Convert one class-specific dataframe into project-style NPZ arrays."""
    scaler = MinMaxScaler()
    train_features = train_df[FEATURE_COLUMNS].astype(np.float32).values
    val_features = val_df[FEATURE_COLUMNS].astype(np.float32).values
    test_features = test_df[FEATURE_COLUMNS].astype(np.float32).values

    if renormalize:
        train_features = scaler.fit_transform(train_features).astype(np.float32)
        val_features = scaler.transform(val_features).astype(np.float32)
        test_features = scaler.transform(test_features).astype(np.float32)

    x_train = np.expand_dims(train_features, axis=1)
    x_val = np.expand_dims(val_features, axis=1)
    x_test = np.expand_dims(test_features, axis=1)

    y_train = np.full(len(x_train), label_id, dtype=np.int64)
    y_val = np.full(len(x_val), label_id, dtype=np.int64)
    y_test = np.full(len(x_test), label_id, dtype=np.int64)
    return SplitBundle(x_train, y_train, x_val, y_val, x_test, y_test)


def main() -> None:
    """Run STI preprocessing and export train/val/test NPZ files."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bsdtar_path = require_bsdtar()

    train_parts: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    val_labels: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    metadata: dict[str, object] = {
        "dataset_name": "STI",
        "input_shape": [1, len(FEATURE_COLUMNS)],
        "num_features": len(FEATURE_COLUMNS),
        "num_classes": len(DATASET_FILES),
        "feature_columns": FEATURE_COLUMNS,
        "splits": {},
        "label_mapping": {},
        "source_dir": args.sti_dir,
        "renormalized": bool(args.renormalize),
    }

    print("=" * 60)
    print("Module: STI Dataset Preparation")
    print("=" * 60)

    for rar_name, csv_name, label_name, label_id in DATASET_FILES:
        rar_path = os.path.join(args.sti_dir, rar_name)
        if not os.path.exists(rar_path):
            raise FileNotFoundError(f"Missing STI archive: {rar_path}")

        frame = read_csv_from_rar(bsdtar_path, rar_path, csv_name)
        if "Label" not in frame.columns:
            raise RuntimeError(f"Missing Label column in {csv_name}")
        if any(feature not in frame.columns for feature in FEATURE_COLUMNS):
            missing = [feature for feature in FEATURE_COLUMNS if feature not in frame.columns]
            raise RuntimeError(f"Missing expected STI feature columns in {csv_name}: {missing}")

        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
        train_df, val_df, test_df = split_frame(frame, args.train_ratio, args.val_ratio)
        bundle = encode_and_stack(train_df, val_df, test_df, label_id, args.renormalize)

        train_parts.append(bundle.x_train)
        train_labels.append(bundle.y_train)
        val_parts.append(bundle.x_val)
        val_labels.append(bundle.y_val)
        test_parts.append(bundle.x_test)
        test_labels.append(bundle.y_test)

        metadata["label_mapping"][str(label_id)] = label_name
        metadata["splits"][label_name] = {
            "train": int(len(bundle.x_train)),
            "val": int(len(bundle.x_val)),
            "test": int(len(bundle.x_test)),
            "total": int(len(frame)),
        }
        print(
            f"[{label_id}] {label_name:<18} "
            f"train={len(bundle.x_train):>7} "
            f"val={len(bundle.x_val):>7} "
            f"test={len(bundle.x_test):>7}"
        )

    x_train = np.concatenate(train_parts, axis=0).astype(np.float32)
    y_train = np.concatenate(train_labels, axis=0).astype(np.int64)
    x_val = np.concatenate(val_parts, axis=0).astype(np.float32)
    y_val = np.concatenate(val_labels, axis=0).astype(np.int64)
    x_test = np.concatenate(test_parts, axis=0).astype(np.float32)
    y_test = np.concatenate(test_labels, axis=0).astype(np.int64)

    np.savez(os.path.join(args.output_dir, "train.npz"), X=x_train, y=y_train)
    np.savez(os.path.join(args.output_dir, "val.npz"), X=x_val, y=y_val)
    np.savez(os.path.join(args.output_dir, "test.npz"), X=x_test, y=y_test)

    metadata["global_shapes"] = {
        "train": list(x_train.shape),
        "val": list(x_val.shape),
        "test": list(x_test.shape),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"Saved train.npz to {args.output_dir}")
    print(f"Train shape: {x_train.shape}  Val shape: {x_val.shape}  Test shape: {x_test.shape}")
    print("Feature mode: tabular STI rows exported as sequence length 1")


if __name__ == "__main__":
    main()
