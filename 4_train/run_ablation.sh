#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/lithic/final/ns3-gpu-venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "GPU venv not found: ${VENV_PYTHON}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

exec "${VENV_PYTHON}" scripts/run_ablation.py \
  --data_dir ../dataset_cicids17 \
  --device cuda \
  "$@"
