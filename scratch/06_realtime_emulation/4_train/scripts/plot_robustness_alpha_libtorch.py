#!/usr/bin/env python3
"""Plot robustness experiment results: alpha sweep for Level 4B."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = TRAIN_ROOT / "experiments" / "OrbitShield_FL_ns3_libtorch" / "alpha_sweep"
PLOT_DIR = OUTPUT_BASE / "plots"
ALPHAS = [0.05, 0.1, 0.3, 0.5, 1.0, 5.0]

# Baseline: single-machine test accuracy from official Level 4B result
BASELINE_ACC = 0.960957


def load_round_metrics(alpha: float) -> pd.DataFrame | None:
    alpha_tag = str(alpha).replace(".", "p")
    csv_path = OUTPUT_BASE / f"alpha_{alpha_tag}" / "round_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def load_summary(alpha: float) -> dict:
    alpha_tag = str(alpha).replace(".", "p")
    p = OUTPUT_BASE / f"alpha_{alpha_tag}" / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def plot_alpha_vs_accuracy(summary_csv: Path) -> None:
    if not summary_csv.exists():
        print(f"[skip] {summary_csv} not found, skipping alpha_vs_accuracy plot.")
        return

    df = pd.read_csv(summary_csv)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["alpha"], df["test_accuracy"], "o-", color="#c0392b", linewidth=2.5,
            markersize=8, label="OrbitShield_FL (Level 4B)")
    ax.axhline(BASELINE_ACC, color="gray", linestyle="--", linewidth=1.5,
               label=f"Single-machine baseline ({BASELINE_ACC:.4f})")

    for _, row in df.iterrows():
        ax.annotate(f"{row['test_accuracy']:.4f}",
                    xy=(row["alpha"], row["test_accuracy"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=11)

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet α (Data Heterogeneity)", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title("OrbitShield_FL (Level 4B) Robustness under Different Data Heterogeneity", fontsize=13)
    ax.legend(fontsize=12)
    ax.set_ylim(0.80, 1.01)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=11)

    out = PLOT_DIR / "alpha_vs_accuracy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_convergence_curves() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Color gradient: dark blue (small alpha / high heterogeneity) -> light blue (large alpha / IID)
    cmap = plt.cm.Blues
    color_vals = np.linspace(0.4, 0.9, len(ALPHAS))

    fig, ax = plt.subplots(figsize=(12, 7))
    plotted = False
    for i, alpha in enumerate(ALPHAS):
        df = load_round_metrics(alpha)
        if df is None or "val_accuracy" not in df.columns:
            print(f"[warn] no round_metrics for alpha={alpha}, skipping.")
            continue
        color = cmap(color_vals[i])
        label = f"α={alpha} ({'high het.' if alpha <= 0.1 else ('IID-like' if alpha >= 1.0 else 'moderate')})"
        ax.plot(df["round"], df["val_accuracy"], "o-", color=color,
                linewidth=2, markersize=5, label=label)
        plotted = True

    if not plotted:
        print("[warn] No convergence data found, skipping convergence curves plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Federated Round", fontsize=13)
    ax.set_ylabel("Validation Accuracy", fontsize=13)
    ax.set_title("OrbitShield_FL (Level 4B) Convergence under Different Data Heterogeneity", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0.80, 1.01)
    ax.grid(True, alpha=0.3)

    out = PLOT_DIR / "alpha_convergence_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    summary_csv = OUTPUT_BASE / "alpha_sweep_summary.csv"
    plot_alpha_vs_accuracy(summary_csv)
    plot_convergence_curves()


if __name__ == "__main__":
    main()
