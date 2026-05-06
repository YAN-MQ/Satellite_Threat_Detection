#!/usr/bin/env python3
"""Generate a t-SNE figure for the selected GRU experiment."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset_profiles import get_dataset_profile
from src.data import load_npz_data
from src.models import DSC_CBAM_GRU

CLASS_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE embeddings for the selected dataset")
    parser.add_argument("--dataset", choices=["cicids17", "sti"], default="cicids17")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_dir", default="experiments/visualization")
    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def extract_embeddings(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    captured = []

    def hook_fn(module, inputs, output):
        captured.append(output.detach().cpu().numpy())

    hook = model.fc[0].register_forward_hook(hook_fn)
    with torch.no_grad():
        tensor = torch.from_numpy(x).float().to(device)
        _ = model(tensor)
    hook.remove()

    if not captured:
        raise RuntimeError("Failed to capture embeddings from model.fc[0]")
    return captured[0]


def main() -> None:
    args = parse_args()
    profile = get_dataset_profile(args.dataset)
    data_dir = args.data_dir if args.data_dir else os.path.join(PROJECT_DIR, profile.data_dir)
    model_path = args.model_path if args.model_path else os.path.join(PROJECT_DIR, profile.output_checkpoint)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_DIR, args.output_dir)
    input_dim = args.input_dim or profile.input_dim
    num_classes = args.num_classes or profile.num_classes

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    _, _, _, _, x_test, y_test = load_npz_data(data_dir)
    sample_count = min(args.n_samples, len(x_test))
    indices = np.random.default_rng(42).choice(len(x_test), size=sample_count, replace=False)
    x_subset = x_test[indices]
    y_subset = y_test[indices]
    perplexity = min(float(args.perplexity), float(max(1, sample_count - 1)))

    model = DSC_CBAM_GRU(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    )
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    embeddings = extract_embeddings(model, x_subset, device)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    embedded = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for class_id, class_name in enumerate(profile.class_names[:num_classes]):
        mask = y_subset == class_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=18,
            alpha=0.65,
            c=CLASS_COLORS[class_id % len(CLASS_COLORS)],
            label=class_name,
        )

    ax.set_title("t-SNE of DSC-CBAM-GRU Latent Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()

    figure_path = os.path.join(output_dir, f"tsne_{profile.name}_gru.png")
    data_path = os.path.join(output_dir, f"tsne_{profile.name}_gru.npz")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    np.savez(data_path, embedding=embedded, labels=y_subset, indices=indices)
    print(f"Saved t-SNE figure to {figure_path}")
    print(f"Saved embedding data to {data_path}")


if __name__ == "__main__":
    main()
