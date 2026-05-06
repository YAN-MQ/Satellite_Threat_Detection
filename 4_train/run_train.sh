#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/lithic/final/ns3-gpu-venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "GPU venv not found: ${VENV_PYTHON}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

exec "${VENV_PYTHON}" scripts/train_gru.py \
  --dataset cicids17 \
  --device cuda \
  --epochs 100 \
  --batch_size 128 \
  --hidden_dim 32 \
  --no-bidirectional \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48 \
  --lr 0.0003 \
  --weight_decay 0.01 \
  --early_stopping_patience 10 \
  --output_dir checkpoints_gru_formal_tuned \
  "$@"
