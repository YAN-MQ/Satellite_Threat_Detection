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
  "${PYTHON_BIN}" scripts/train_federated_window.py \
    --data_dir ../dataset_window \
    --num_clients 12 \
    --num_planes 3 \
    --rounds 20 \
    --local_epochs 1 \
    --batch_size 512 \
    --partition_mode dirichlet \
    --dirichlet_alpha 0.3 \
    --beta 0.1 \
    --lambda_s 0.1 \
    --rho 0.5 \
    --mu 0.8 \
    --global_momentum 0.1 \
    --warmup_rounds 2 \
    --method "${METHOD}" \
    --init_checkpoint checkpoints_gru/window_gru_best.pt \
    --device cuda \
    --output_dir "experiments_window/federated/${METHOD}"
done
