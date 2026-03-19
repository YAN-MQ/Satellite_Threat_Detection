#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
PYTHON_BIN="/home/lithic/final/ns3-gpu-venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

echo "Running OrbitShield_FL with the best tuned default configuration"

"${PYTHON_BIN}" scripts/train_federated.py \
  --dataset cicids17 \
  --method full \
  --device cuda
