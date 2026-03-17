#!/usr/bin/env python3
"""Extract 18 packet-window features from captured PCAPs."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, rdpcap
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(BASE_DIR, "captured_window")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_window")
WINDOW = 10
STRIDE = 1
FILES = [("benign.pcap", 0), ("ddos.pcap", 1), ("portscan.pcap", 2)]
FEATURES = [
    "IAT",
    "size",
    "proto",
    "SYN",
    "ACK",
    "RST",
    "FIN",
    "PSH",
    "Size_M",
    "Size_S",
    "IAT_M",
    "IAT_S",
    "IAT_X",
    "IAT_N",
    "PPS",
    "BPS",
    "SYN_R",
    "ACK_R",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sliding-window packet features")
    parser.add_argument("--capture-dir", default=CAPTURE_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()


def parse_packet(pkt):
    if IP not in pkt:
        return None
    row = {
        "ts": float(pkt.time),
        "size": len(pkt),
        "proto": pkt[IP].proto,
        "SYN": 0,
        "ACK": 0,
        "RST": 0,
        "FIN": 0,
        "PSH": 0,
    }
    if TCP in pkt:
        flags = str(pkt[TCP].flags)
        row.update(
            {
                "SYN": int("S" in flags),
                "ACK": int("A" in flags),
                "RST": int("R" in flags),
                "FIN": int("F" in flags),
                "PSH": int("P" in flags),
            }
        )
    return row


def process_pcap(path: str) -> pd.DataFrame:
    packets = rdpcap(path)
    rows = []
    prev_ts = None
    for pkt in packets:
        row = parse_packet(pkt)
        if row is None:
            continue
        row["IAT"] = 0 if prev_ts is None else row["ts"] - prev_ts
        prev_ts = row["ts"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    w = WINDOW
    df["Size_M"] = df["size"].rolling(w, min_periods=1).mean()
    df["Size_S"] = df["size"].rolling(w, min_periods=1).std().fillna(0)
    df["IAT_M"] = df["IAT"].rolling(w, min_periods=1).mean()
    df["IAT_S"] = df["IAT"].rolling(w, min_periods=1).std().fillna(0)
    df["IAT_X"] = df["IAT"].rolling(w, min_periods=1).max()
    df["IAT_N"] = df["IAT"].rolling(w, min_periods=1).min()

    span = df["ts"].rolling(w, min_periods=1).apply(lambda x: x.max() - x.min() if len(x) > 1 else 1.0)
    pkt_count = df["size"].rolling(w, min_periods=1).count()
    byte_sum = df["size"].rolling(w, min_periods=1).sum()
    df["PPS"] = pkt_count / (span + 1e-8)
    df["BPS"] = byte_sum / (span + 1e-8)
    df["SYN_R"] = df["SYN"].rolling(w, min_periods=1).sum() / (pkt_count + 1e-8)
    df["ACK_R"] = df["ACK"].rolling(w, min_periods=1).sum() / (pkt_count + 1e-8)
    return df.fillna(0).replace([np.inf, -np.inf], 0)


def split_frame(df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def make_sequences(values: np.ndarray) -> np.ndarray:
    return np.array([values[i : i + WINDOW] for i in range(0, len(values) - WINDOW + 1, STRIDE)], dtype=np.float32)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_train, all_val, all_test = [], [], []
    print("=" * 60)
    print("Module 3: Window Feature Extraction")
    print("=" * 60)

    for filename, label in FILES:
        path = os.path.join(args.capture_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing capture file: {path}")

        print(f"\n[{label}] {filename}")
        df = process_pcap(path)
        if df.empty:
            raise RuntimeError(f"No IP packets extracted from {path}")

        train_df, val_df, test_df = split_frame(df)
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[FEATURES])
        val_scaled = scaler.transform(val_df[FEATURES])
        test_scaled = scaler.transform(test_df[FEATURES])

        x_train = make_sequences(train_scaled)
        x_val = make_sequences(val_scaled)
        x_test = make_sequences(test_scaled)
        y_train = np.full(len(x_train), label, dtype=np.int64)
        y_val = np.full(len(x_val), label, dtype=np.int64)
        y_test = np.full(len(x_test), label, dtype=np.int64)

        print(f"  Train {x_train.shape}  Val {x_val.shape}  Test {x_test.shape}")
        all_train.append((x_train, y_train))
        all_val.append((x_val, y_val))
        all_test.append((x_test, y_test))

    x_train = np.concatenate([item[0] for item in all_train])
    y_train = np.concatenate([item[1] for item in all_train])
    x_val = np.concatenate([item[0] for item in all_val])
    y_val = np.concatenate([item[1] for item in all_val])
    x_test = np.concatenate([item[0] for item in all_test])
    y_test = np.concatenate([item[1] for item in all_test])

    np.savez(os.path.join(args.output_dir, "train.npz"), X=x_train, y=y_train)
    np.savez(os.path.join(args.output_dir, "val.npz"), X=x_val, y=y_val)
    np.savez(os.path.join(args.output_dir, "test.npz"), X=x_test, y=y_test)
    print(f"\nSaved dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
