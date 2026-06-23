#!/usr/bin/env bash
# 启动单个 Flower client (真机部署用, 在树莓派上跑).
#
# 用法:  ./scripts/run_client.sh <config.yaml> <cid> [server_addr]
# 例:    ./scripts/run_client.sh configs/mnist_iid_2clients.yaml 0 192.168.1.100:8080
#
# 注意:
#   - server_addr 可省略, 此时使用 cfg 里的 server_address
#   - cid 必须 < cfg.num_clients
#   - 树莓派上请先激活 .venv: source .venv/bin/activate
set -euo pipefail

CFG=${1:?usage: $0 <config.yaml> <cid> [server_addr]}
CID=${2:?usage: $0 <config.yaml> <cid> [server_addr]}
ADDR=${3:-}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

if [[ -n "$ADDR" ]]; then
  python -m fed_pi.client --config "$CFG" --cid "$CID" --server_address "$ADDR"
else
  python -m fed_pi.client --config "$CFG" --cid "$CID"
fi
