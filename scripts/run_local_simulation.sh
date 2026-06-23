#!/usr/bin/env bash
# 单机多进程模拟联邦学习 (开发/调试用).
#
# 用法:  ./scripts/run_local_simulation.sh configs/mnist_iid_2clients.yaml
#
# 步骤:
#   1. 解析 cfg 里的 num_clients
#   2. 后台启动 server, 等它绑定端口
#   3. 后台启动 N 个 client, 各传 --cid
#   4. wait 所有进程, Ctrl+C 时清理
set -euo pipefail

CFG=${1:?usage: $0 <config.yaml>}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

if [[ ! -f "$CFG" ]]; then
  echo "config not found: $CFG" >&2
  exit 1
fi

# 用 Python 解析 yaml 取出 num_clients (避免引入 yq 依赖)
NUM_CLIENTS=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['num_clients'])")
EXP_NAME=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['exp_name'])")

LOG_DIR="results/$EXP_NAME/logs"
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  echo
  echo "[run_local] cleaning up ${#pids[@]} processes..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- server ---
echo "[run_local] starting server (cfg=$CFG, exp=$EXP_NAME, clients=$NUM_CLIENTS)"
python -m fed_pi.server --config "$CFG" 2>&1 | tee "$LOG_DIR/server.log" &
pids+=($!)

# 等 server 绑定端口 (粗暴 sleep, 课程演示足够)
sleep 5

# --- clients ---
for ((cid=0; cid<NUM_CLIENTS; cid++)); do
  echo "[run_local] starting client cid=$cid"
  python -m fed_pi.client --config "$CFG" --cid "$cid" \
    2>&1 | tee "$LOG_DIR/client_$cid.log" &
  pids+=($!)
done

# 等 server 自然退出 (跑完 num_rounds 后 start_server 返回, 子进程会退出)
wait "${pids[0]}"
echo "[run_local] server finished. exp results: results/$EXP_NAME/"
