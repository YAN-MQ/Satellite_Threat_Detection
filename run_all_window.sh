#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
NS3_DIR="$(cd "$BASE_DIR/../.." && pwd)"
FRAG_DIR="$BASE_DIR/fragments_window"
CAPTURE_DIR="$BASE_DIR/captured_window"
DATASET_OUT="$BASE_DIR/dataset_cicids17"
MAX_PACKETS="${MAX_PACKETS:-50000}"

echo "=========================================="
echo "LEO Satellite Window Pipeline"
echo "=========================================="

cd "$BASE_DIR"

echo "[1/5] Extracting windows and fragmenting packets"
python3 1_fragment_pcap_window.py --max-packets "$MAX_PACKETS"

echo "[2/5] Building ns-3 scratch target"
cd "$NS3_DIR"
cmake --build build --target scratch_06_realtime_emulation_realtime_satellite -j"$(nproc)"

BIN=$(find "$NS3_DIR/build/scratch/06_realtime_emulation" -maxdepth 1 -type f -name 'ns3.46.1-realtime_satellite-*' | head -n 1)
if [ ! -x "$BIN" ]; then
  echo "Built target is missing or not executable: $BIN"
  exit 1
fi

echo "[3/5] Creating TAP interfaces"
ip tuntap add dev tap-left mode tap 2>/dev/null || true
ip tuntap add dev tap-right mode tap 2>/dev/null || true
ip link set tap-left up
ip link set tap-right up

echo "[4/5] Running emulation and capturing traffic"
mkdir -p "$CAPTURE_DIR"
rm -f "$CAPTURE_DIR"/benign.pcap "$CAPTURE_DIR"/ddos.pcap "$CAPTURE_DIR"/portscan.pcap "$CAPTURE_DIR"/dos.pcap

"$BIN" --time=120 > /tmp/realtime_satellite.log 2>&1 &
NS3_PID=$!
sleep 3

for traffic in benign ddos portscan; do
  echo "  replaying $traffic"
  tcpdump -i tap-right -U -w "$CAPTURE_DIR/$traffic.pcap" >/tmp/tcpdump_${traffic}.log 2>&1 &
  TCPDUMP_PID=$!
  sleep 1
  tcpreplay -i tap-left -M 100 "$FRAG_DIR/$traffic.pcap" >/tmp/tcpreplay_${traffic}.log 2>&1
  sleep 3
  kill "$TCPDUMP_PID" 2>/dev/null || true
  wait "$TCPDUMP_PID" 2>/dev/null || true
done

kill "$NS3_PID" 2>/dev/null || true
wait "$NS3_PID" 2>/dev/null || true

echo "[5/5] Extracting features and training GRU"
cd "$BASE_DIR"
python3 2_extract_features_window.py
cd 4_train
python3 scripts/train_gru.py --data_dir ../dataset_cicids17 --epochs 20 --num_classes 3 --input_dim 18

echo "=========================================="
echo "Pipeline Complete"
echo "=========================================="
