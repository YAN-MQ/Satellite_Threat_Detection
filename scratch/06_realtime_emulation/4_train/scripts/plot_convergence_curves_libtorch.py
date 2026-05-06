#!/usr/bin/env python3
"""Plot federated convergence curves comparing Level 4B, Level 3, Level 1, and baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = TRAIN_ROOT / "experiments"
PLOT_DIR = EXP_ROOT / "visualization"

# Curve definitions: (label, csv_path, color, linestyle, linewidth, is_hero)
CURVES = [
    (
        "Level 4B: OrbitShield_FL (NS-3+libtorch)",
        EXP_ROOT / "OrbitShield_FL_ns3_libtorch" / "cicids17" / "round_metrics.csv",
        "#c0392b", "-", 3.0, True,
    ),
    (
        "Level 3: OrbitShield_FL (NS-3 online)",
        EXP_ROOT / "OrbitShield_FL_ns3_online" / "cicids17" / "round_metrics.csv",
        "#e67e22", "-", 2.0, False,
    ),
    (
        "Level 2: OrbitShield_FL (NS-3 offline)",
        EXP_ROOT / "OrbitShield_FL_ns3" / "cicids17" / "round_metrics.csv",
        "#8e44ad", "-", 2.0, False,
    ),
    (
        "Level 1: OrbitShield_FL (heuristic)",
        EXP_ROOT / "OrbitShield_FL" / "cicids17" / "round_metrics.csv",
        "#2980b9", "-", 2.0, False,
    ),
    (
        "FedAvg",
        EXP_ROOT / "OrbitShield_FL" / "baselines" / "fedavg" / "round_metrics.csv",
        "#7f8c8d", "--", 1.8, False,
    ),
    (
        "Intra-only",
        EXP_ROOT / "OrbitShield_FL" / "baselines" / "intra_only" / "round_metrics.csv",
        "#27ae60", "--", 1.8, False,
    ),
]


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "val_accuracy" not in df.columns or df["val_accuracy"].isna().all():
            return None
        return df
    except Exception:
        return None


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    plotted_any = False

    for label, csv_path, color, ls, lw, is_hero in CURVES:
        df = load_csv(csv_path)
        if df is None:
            print(f"[skip] {label}: no data at {csv_path}")
            continue

        rounds = df["round"]
        val_acc = df["val_accuracy"]
        marker = "o" if is_hero else "s"
        ms = 7 if is_hero else 5
        ax.plot(rounds, val_acc, marker=marker, color=color, linestyle=ls,
                linewidth=lw, markersize=ms, label=label)

        # Annotate final value
        final_round = rounds.iloc[-1]
        final_acc = val_acc.iloc[-1]
        offset = 8 if is_hero else 5
        ax.annotate(f"{final_acc:.4f}",
                    xy=(final_round, final_acc),
                    xytext=(4, offset), textcoords="offset points",
                    fontsize=10, color=color,
                    fontweight="bold" if is_hero else "normal")
        plotted_any = True
        print(f"  {label}: {len(df)} rounds, final val_acc={final_acc:.4f}")

    if not plotted_any:
        print("[error] No data found for any curve.")
        plt.close(fig)
        return

    ax.set_xlabel("Federated Round", fontsize=13)
    ax.set_ylabel("Validation Accuracy", fontsize=13)
    ax.set_title("Federated Learning Convergence Comparison (CICIDS17)", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0.82, 1.01)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    out = PLOT_DIR / "fl_convergence_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
