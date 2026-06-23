#!/usr/bin/env bash
# 在 Mac (server) + pi-01 + pi-02 (clients) 上跑 5 个 MNIST 真机实验.
# 用法: ./scripts/run_pi_mnist_experiments.sh
#
# 注意:
#   - server 用 8082 端口 (避免与本机 CIFAR 占的 8081 冲突)
#   - 2 台 Pi 通过 pi_ssh.exp / pi02_ssh.exp 自动 ssh
#   - 每个实验都自动等 Pi client 完成 (server 退出后 Pi 进程自动结束)
set -uo pipefail
# 注意: 不用 set -e — expect 调用偶尔会返回非 0, 但远程 nohup 命令已经在 Pi 后台跑起来了

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)/scripts"

# 必须先激活 venv: 后面用 python 解析 yaml + 启动 server
source .venv/bin/activate

MAC_IP="${MAC_IP:-192.168.18.34}"
PORT=8082

CONFIGS=(
  configs/mnist_iid_2clients.yaml      # E1
  configs/mnist_noniid_a01.yaml        # E2
  configs/mnist_noniid_a05.yaml        # E3
  configs/mnist_noniid_a10.yaml        # E4
  configs/mnist_imbalance_91.yaml      # E5
)

PI_SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o PreferredAuthentications=password -o PubkeyAuthentication=no \
    -o ConnectTimeout=15"

# 把 server.py 的最新版同步到两台 Pi (server 这次加了 --server_address 支持)
echo "[mnist-pi] syncing latest fed_pi/server.py and configs to both Pis..."
for host in pi-01.local pi-02.local; do
  rsync -az -e "ssh $PI_SSH_OPTS" \
    fed_pi/server.py "pi@$host:~/fed_pi/fed_pi/server.py" 2>&1 | tail -1
  rsync -az -e "ssh $PI_SSH_OPTS" \
    configs/ "pi@$host:~/fed_pi/configs/" 2>&1 | tail -1
done

run_one_experiment() {
  local cfg=$1
  local exp_name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['exp_name'])")
  local n_clients=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['num_clients'])")

  echo ""
  echo "================================================================"
  echo "[mnist-pi] starting experiment: $exp_name (clients=$n_clients) at $(date +%H:%M:%S)"
  echo "================================================================"

  # 1) 启动 Mac server (后台)
  nohup python -m fed_pi.server --config "$cfg" --server_address "0.0.0.0:$PORT" \
      > "/tmp/fedpi_server_${exp_name}.log" 2>&1 &
  local SERVER_PID=$!
  echo "[mnist-pi] server started pid=$SERVER_PID"
  sleep 3

  # 2) 启动 Pi client(s) 在后台 (用 nohup + & 在 Pi 上)
  if [ "$n_clients" -ge 1 ]; then
    echo "[mnist-pi] launching client on pi-01 (cid=0)..."
    "$SCRIPT_DIR/pi_ssh.exp" "cd ~/fed_pi && nohup bash -c 'source .venv/bin/activate && python -m fed_pi.client --config $cfg --cid 0 --server_address ${MAC_IP}:${PORT} > ~/fed_pi/_client.log 2>&1' >/dev/null 2>&1 & echo pi01_client_pid=\$!" 2>&1 | tail -2
  fi
  if [ "$n_clients" -ge 2 ]; then
    echo "[mnist-pi] launching client on pi-02 (cid=1)..."
    "$SCRIPT_DIR/pi02_ssh.exp" "cd ~/fed_pi && nohup bash -c 'source .venv/bin/activate && python -m fed_pi.client --config $cfg --cid 1 --server_address ${MAC_IP}:${PORT} > ~/fed_pi/_client.log 2>&1' >/dev/null 2>&1 & echo pi02_client_pid=\$!" 2>&1 | tail -2
  fi

  # 3) wait server 结束
  echo "[mnist-pi] waiting for server pid=$SERVER_PID to finish..."
  wait $SERVER_PID || true

  # 4) 给 Pi client 一点时间完成清理
  sleep 5
  echo "[mnist-pi] experiment $exp_name DONE at $(date +%H:%M:%S)"
}

for cfg in "${CONFIGS[@]}"; do
  run_one_experiment "$cfg"
done

echo ""
echo "================================================================"
echo "[mnist-pi] ALL 5 MNIST EXPERIMENTS DONE at $(date +%H:%M:%S)"
echo "================================================================"
ls -la results/
