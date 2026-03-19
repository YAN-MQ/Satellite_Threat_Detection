#!/usr/bin/env python3
"""Train the DSC-CBAM-GRU classifier."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_dataloaders, load_npz_data
from src.models import DSC_CBAM_GRU, count_flops, count_parameters
from src.training import Trainer, get_optimizer, get_scheduler
from dataset_profiles import get_dataset_profile
from experiment_utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DSC-CBAM-GRU on CICIDS17 or STI")
    parser.add_argument("--dataset", choices=["cicids17", "sti"], default="cicids17")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_dataset_profile(args.dataset)
    data_dir = args.data_dir or profile.data_dir
    input_dim = args.input_dim or profile.input_dim
    num_classes = args.num_classes or profile.num_classes
    seq_len = profile.seq_len
    output_path = args.output or profile.output_checkpoint
    device = resolve_device(args.device)

    print("=" * 60)
    print("DSC-CBAM-GRU Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {profile.name}")
    print(f"Data dir: {data_dir}")

    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(data_dir)
    if args.max_samples:
        rng = np.random.default_rng(42)

        def subsample(features, labels):
            limit = min(args.max_samples, len(features))
            indices = rng.choice(len(features), size=limit, replace=False)
            return features[indices], labels[indices]

        x_train, y_train = subsample(x_train, y_train)
        x_val, y_val = subsample(x_val, y_val)
        x_test, y_test = subsample(x_test, y_test)

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    train_loader, val_loader, test_loader = create_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    model = DSC_CBAM_GRU(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    )
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Approx FLOPs: {count_flops(model, input_size=(1, seq_len, input_dim)):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name="adamw", lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_name="plateau")
    trainer = Trainer(model, criterion, optimizer, scheduler, device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trainer.train(train_loader, val_loader, num_epochs=args.epochs, save_path=output_path)
    metrics = trainer.evaluate(test_loader)

    print("\nTest metrics")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
