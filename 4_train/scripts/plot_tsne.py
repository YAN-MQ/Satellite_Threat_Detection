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

from src.data import load_npz_data
from src.models import DSC_CBAM_GRU

CLASS_NAMES = ["Benign", "DDoS", "PortScan"]
CLASS_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE embeddings for the selected dataset")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--model_path", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/visualization")
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
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
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    _, _, _, _, x_test, y_test = load_npz_data(args.data_dir)
    sample_count = min(args.n_samples, len(x_test))
    indices = np.random.default_rng(42).choice(len(x_test), size=sample_count, replace=False)
    x_subset = x_test[indices]
    y_subset = y_test[indices]
    perplexity = min(float(args.perplexity), float(max(1, sample_count - 1)))

    model = DSC_CBAM_GRU(input_dim=args.input_dim, num_classes=args.num_classes)
    state_dict = torch.load(args.model_path, map_location=device)
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
    for class_id, class_name in enumerate(CLASS_NAMES[: args.num_classes]):
        mask = y_subset == class_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=18,
            alpha=0.65,
            c=CLASS_COLORS[class_id],
            label=class_name,
        )

    ax.set_title("t-SNE of DSC-CBAM-GRU Latent Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()

    figure_path = os.path.join(args.output_dir, "tsne_cicids17_gru.png")
    data_path = os.path.join(args.output_dir, "tsne_cicids17_gru.npz")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    np.savez(data_path, embedding=embedded, labels=y_subset, indices=indices)
    print(f"Saved t-SNE figure to {figure_path}")
    print(f"Saved embedding data to {data_path}")


if __name__ == "__main__":
    main()
