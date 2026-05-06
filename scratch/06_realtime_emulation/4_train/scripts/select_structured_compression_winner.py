#!/usr/bin/env python3
"""Select the formal structured compression winner from candidate summaries."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


TARGETS = {
    "parameter_reduction_ratio": 0.25,
    "latency_reduction_ratio": 0.20,
    "accuracy_drop_max": 0.003,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select structured compression winner.")
    parser.add_argument(
        "--candidates_dir",
        default="experiments/compression/structured_candidates_formal_tuned",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_passes(candidate: dict[str, Any]) -> dict[str, bool]:
    return {
        "parameter_reduction_ratio": bool(candidate["parameter_reduction_ratio"] >= TARGETS["parameter_reduction_ratio"]),
        "latency_reduction_ratio": bool(candidate["latency_reduction_ratio"] >= TARGETS["latency_reduction_ratio"]),
        "accuracy_drop_vs_formal": bool(candidate["accuracy_drop_vs_formal"] <= TARGETS["accuracy_drop_max"]),
    }


def choose_winner(candidates: dict[str, dict[str, Any]]) -> str | None:
    passing = []
    for name, candidate in candidates.items():
        passes = candidate_passes(candidate)
        candidate["passes_thresholds"] = passes
        candidate["meets_all_targets"] = bool(all(passes.values()))
        if candidate["meets_all_targets"]:
            passing.append((name, candidate))

    if not passing:
        return None

    passing.sort(
        key=lambda item: (
            float(item[1]["accuracy"]),
            -float(item[1]["int8_cpu_latency_ms"]),
        ),
        reverse=True,
    )
    return passing[0][0]


def build_final_summary(candidates_dir: str) -> dict[str, Any]:
    aggregate_path = os.path.join(candidates_dir, "all_candidates_summary.json")
    if os.path.exists(aggregate_path):
        aggregate = load_json(aggregate_path)
        formal_baseline = aggregate["formal_tuned_baseline"]
        candidates = aggregate["candidates"]
    else:
        candidates = {}
        for candidate_name in ["A1", "A2", "A3"]:
            summary_path = os.path.join(candidates_dir, candidate_name, "candidate_summary.json")
            if not os.path.exists(summary_path):
                raise FileNotFoundError(f"Missing candidate summary: {summary_path}")
            candidates[candidate_name] = load_json(summary_path)
        first_candidate = next(iter(candidates.values()))
        formal_baseline = first_candidate["baseline_formal"]

    winner = choose_winner(candidates)
    final_summary = {
        "formal_tuned_baseline": formal_baseline,
        "candidates": candidates,
        "winner": winner,
        "winner_meets_all_targets": bool(winner is not None),
        "targets": TARGETS,
    }
    return final_summary


def main() -> None:
    args = parse_args()
    candidates_dir = args.candidates_dir
    if not os.path.isabs(candidates_dir):
        candidates_dir = os.path.abspath(candidates_dir)

    final_summary = build_final_summary(candidates_dir)
    output_path = os.path.join(candidates_dir, "structured_formal_summary.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2, ensure_ascii=False)

    if final_summary["winner"] is None:
        print("No candidate meets all thresholds")
    else:
        print(f"Winner: {final_summary['winner']}")
    print(f"Wrote summary: {output_path}")


if __name__ == "__main__":
    main()
