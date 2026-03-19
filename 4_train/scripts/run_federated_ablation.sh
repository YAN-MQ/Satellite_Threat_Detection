#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
PYTHON_BIN="/home/lithic/final/ns3-gpu-venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

for METHOD in fedavg intra_only intra_gossip full; do
  if [[ "${METHOD}" == "full" ]]; then
    echo "Running OrbitShield_FL baseline"
  else
    echo "Running federated baseline: ${METHOD}"
  fi
  "${PYTHON_BIN}" scripts/train_federated.py \
    --dataset cicids17 \
    --method "${METHOD}" \
    --device cuda \
    --output_dir "experiments/OrbitShield_FL/baselines/${METHOD}"
done
