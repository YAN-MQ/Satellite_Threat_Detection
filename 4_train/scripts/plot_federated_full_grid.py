#!/usr/bin/env python3
"""Plot heatmaps and trend charts for the OrbitShield_FL grid search."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the federated full-grid plotter."""
    parser = argparse.ArgumentParser(description="Plot OrbitShield_FL grid-search results")
    parser.add_argument(
        "--csv_path",
        default="experiments_window/federated/full_grid/full_grid_summary.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments_window/federated/full_grid/plots",
    )
    return parser.parse_args()


def _draw_heatmap(ax, pivot: pd.DataFrame, title: str) -> None:
    """Draw a numeric heatmap without extra plotting dependencies."""
    image = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    ax.set_xlabel("warmup_rounds")
    ax.set_ylabel("beta")
    ax.set_title(title)

    for row_idx, beta in enumerate(pivot.index):
        for col_idx, warmup in enumerate(pivot.columns):
            value = pivot.loc[beta, warmup]
            ax.text(col_idx, row_idx, f"{value:.4f}", ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_heatmaps(df: pd.DataFrame, output_path: str) -> None:
    """Plot test-accuracy heatmaps for each global momentum slice."""
    momenta = sorted(df["global_momentum"].unique())
    fig, axes = plt.subplots(1, len(momenta), figsize=(6 * len(momenta), 5), constrained_layout=True)
    if len(momenta) == 1:
        axes = [axes]

    for ax, momentum in zip(axes, momenta):
        subset = df[df["global_momentum"] == momentum]
        pivot = subset.pivot(index="beta", columns="warmup_rounds", values="test_accuracy").sort_index().sort_index(axis=1)
        _draw_heatmap(ax, pivot, f"OrbitShield_FL Accuracy Heatmap\n(global_momentum={momentum})")

    fig.suptitle("OrbitShield_FL Grid Search", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_trends(df: pd.DataFrame, output_path: str) -> None:
    """Plot accuracy trends versus warmup rounds for each beta and momentum."""
    momenta = sorted(df["global_momentum"].unique())
    fig, axes = plt.subplots(1, len(momenta), figsize=(6 * len(momenta), 5), sharey=True, constrained_layout=True)
    if len(momenta) == 1:
        axes = [axes]

    for ax, momentum in zip(axes, momenta):
        subset = df[df["global_momentum"] == momentum].sort_values(["beta", "warmup_rounds"])
        for beta, group in subset.groupby("beta"):
            ax.plot(
                group["warmup_rounds"],
                group["test_accuracy"],
                marker="o",
                linewidth=2,
                label=f"beta={beta}",
            )
        ax.set_title(f"OrbitShield_FL Accuracy Trend\n(global_momentum={momentum})")
        ax.set_xlabel("warmup_rounds")
        ax.set_ylabel("test_accuracy")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("OrbitShield_FL Trends", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_best_runs_bar(df: pd.DataFrame, output_path: str, top_k: int = 10) -> None:
    """Plot the top-k runs as a ranked bar chart."""
    top = df.sort_values("test_accuracy", ascending=False).head(top_k).copy()
    labels = [
        f"b={row.beta}\nw={int(row.warmup_rounds)}\ngm={row.global_momentum}"
        for row in top.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = ax.bar(range(len(top)), top["test_accuracy"], color="#c96c33")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("test_accuracy")
    ax.set_title("Top OrbitShield_FL Configurations")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, top["test_accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Load the full-grid CSV and generate visualizations."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    df["beta"] = df["beta"].astype(float)
    df["warmup_rounds"] = df["warmup_rounds"].astype(int)
    df["global_momentum"] = df["global_momentum"].astype(float)
    df["test_accuracy"] = df["test_accuracy"].astype(float)

    plot_heatmaps(df, os.path.join(args.output_dir, "OrbitShield_FL_heatmaps.png"))
    plot_trends(df, os.path.join(args.output_dir, "OrbitShield_FL_trends.png"))
    plot_best_runs_bar(df, os.path.join(args.output_dir, "OrbitShield_FL_top10.png"))

    best = df.sort_values("test_accuracy", ascending=False).iloc[0]
    print("Best OrbitShield_FL configuration")
    print(f"  beta={best['beta']}")
    print(f"  warmup_rounds={int(best['warmup_rounds'])}")
    print(f"  global_momentum={best['global_momentum']}")
    print(f"  test_accuracy={best['test_accuracy']:.6f}")
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
