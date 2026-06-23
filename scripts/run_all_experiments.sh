#!/usr/bin/env bash
# 串行跑全部 8 个实验, 产出 results/*/{metrics.json, curve_*.png}.
#
# 注意: 都用单机模拟模式 (run_local_simulation.sh). 真机部署请手动一次一次跑.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

CONFIGS=(
  configs/mnist_iid_2clients.yaml
  configs/mnist_noniid_a01.yaml
  configs/mnist_noniid_a05.yaml
  configs/mnist_noniid_a10.yaml
  configs/mnist_imbalance_91.yaml
  configs/cifar10_cnn.yaml
  configs/cifar10_mobilenet.yaml
  configs/cifar10_squeezenet.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "========================================"
  echo "running $cfg"
  echo "========================================"
  ./scripts/run_local_simulation.sh "$cfg"
done

echo
echo "all experiments done. generating comparison plots..."
python scripts/plot_comparison.py --group noniid
python scripts/plot_comparison.py --group lightweight
echo "done."
