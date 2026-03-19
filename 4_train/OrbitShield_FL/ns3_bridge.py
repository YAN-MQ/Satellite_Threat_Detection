"""Bridge utilities for loading and validating ns-3 constellation traces."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Ns3RoundTrace:
    """Container for a single round trace exported by federated_constellation.cc."""

    round_idx: int
    payload: dict[str, object]
    path: Path


@dataclass(frozen=True)
class Ns3TraceBundle:
    """Container for a full ns-3 trace directory."""

    trace_dir: Path
    config: dict[str, object]
    manifest: dict[str, object]
    rounds: list[Ns3RoundTrace]


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON file from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_keys(payload: dict[str, object], required_keys: list[str], source: str) -> None:
    """Validate that a JSON payload contains required keys."""
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{source} is missing required keys: {missing}")


def validate_round_trace(payload: dict[str, object], source: str) -> None:
    """Validate the schema of a round trace file."""
    _require_keys(
        payload,
        [
            "schema_version",
            "round",
            "round_duration_s",
            "num_planes",
            "sats_per_plane",
            "planes",
            "satellites",
            "intra_plane_links",
            "inter_plane_links",
            "satellite_links",
        ],
        source,
    )


def load_ns3_trace_bundle(trace_dir: str | Path) -> Ns3TraceBundle:
    """Load constellation config, manifest, and all round traces from a directory."""
    trace_path = Path(trace_dir).resolve()
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_path}")

    config_path = trace_path / "constellation_config.json"
    manifest_path = trace_path / "manifest.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing constellation_config.json in {trace_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {trace_path}")

    config = _load_json(config_path)
    manifest = _load_json(manifest_path)
    _require_keys(config, ["schema_version", "num_planes", "sats_per_plane", "rounds", "static_links"], str(config_path))
    _require_keys(manifest, ["schema_version", "config_file", "round_count", "round_files"], str(manifest_path))

    rounds: list[Ns3RoundTrace] = []
    round_files = manifest["round_files"]
    if not isinstance(round_files, list):
        raise ValueError("manifest.json field round_files must be a list")

    for file_name in round_files:
        round_path = trace_path / str(file_name)
        if not round_path.exists():
            raise FileNotFoundError(f"Missing round trace file referenced by manifest: {round_path}")
        payload = _load_json(round_path)
        validate_round_trace(payload, str(round_path))
        rounds.append(Ns3RoundTrace(round_idx=int(payload["round"]), payload=payload, path=round_path))

    return Ns3TraceBundle(trace_dir=trace_path, config=config, manifest=manifest, rounds=rounds)


def load_ns3_round_trace(trace_dir: str | Path, round_idx: int) -> Ns3RoundTrace:
    """Load a single round trace by round index from an ns-3 trace directory."""
    bundle = load_ns3_trace_bundle(trace_dir)
    for round_trace in bundle.rounds:
        if round_trace.round_idx == round_idx:
            return round_trace
    raise IndexError(f"Round {round_idx} was not found in trace bundle: {bundle.trace_dir}")


def run_federated_constellation(
    binary_path: str | Path,
    output_dir: str | Path,
    *,
    num_planes: int = 3,
    sats_per_plane: int = 4,
    rounds: int = 20,
    round_duration: float = 30.0,
    seed: int = 42,
    extra_args: list[str] | None = None,
) -> Ns3TraceBundle:
    """Run the ns-3 federated constellation simulator and load the generated trace bundle."""
    binary = Path(binary_path).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"Constellation binary not found: {binary}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(binary),
        f"--num-planes={num_planes}",
        f"--sats-per-plane={sats_per_plane}",
        f"--rounds={rounds}",
        f"--round-duration={round_duration}",
        f"--output-dir={out_dir}",
        f"--seed={seed}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)
    return load_ns3_trace_bundle(out_dir)
