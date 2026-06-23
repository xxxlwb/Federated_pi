#!/usr/bin/env bash
# 启动 Flower server (真机部署用).
#
# 用法:  ./scripts/run_server.sh configs/xxx.yaml
#
# 注意:
#   - 默认 cfg 里 server_address 是 "0.0.0.0:8080", 监听所有网卡
#   - 防火墙需要放行 8080 端口
#   - server 会等待 cfg.num_clients 个 client 全部连接才开始训练
set -euo pipefail

CFG=${1:?usage: $0 <config.yaml>}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

python -m fed_pi.server --config "$CFG"
