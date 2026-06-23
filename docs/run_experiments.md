# 实验运行手册 — Mac + 树莓派 真机分布式

> 把 8 个实验在 **Mac (server) + 1 或 2 台 Pi (client)** 上跑起来的逐字命令.
> 每个实验给出 3 种运行方式: 本机模拟 / 1 Pi 真机 / 2 Pi 真机.

## 前提

### Mac 端
- 已建 venv 并装好依赖: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- 知道 Mac 的局域网 IP, 假设为 `192.168.18.34` (本文档示例值)
  - 查 IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`
- 防火墙放行 8081 端口 (macOS 系统设置 → 网络 → 防火墙 → 允许 python 入站)

### 树莓派端
- Pi OS 64-bit, Python 3.11+, 已 ssh 通
- 项目已 rsync 到 `~/fed_pi`, venv 已建好并装完依赖
  ```bash
  ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && python -c 'import torch,flwr; print(torch.__version__, flwr.__version__)'"
  ```
- 知道 Pi 的局域网 IP (Pi-1 假设 `192.168.18.38`, Pi-2 假设 `192.168.18.39`)

### 约定
- 下文 `<MAC_IP>` 代表你 Mac 的实际局域网 IP
- `<PI1>` / `<PI2>` 代表两台 Pi 的 SSH 目标 (主机名或 IP, 例如 `pi@pi-01.local`)
- 所有命令默认在项目根目录 `/Users/luwenbo/projects/test/fed_pi` 执行
- 提交联邦学习实验前, 都需要 `source .venv/bin/activate` 激活 venv

---

## 通用模板

### 模板 A — 本机单机模拟 (开发/调试)
最快, 一个脚本搞定. 适合验证流程或快速出图. **不需要 Pi 参与**.
```bash
# Mac 本机
source .venv/bin/activate
./scripts/run_local_simulation.sh configs/<实验名>.yaml
```

### 模板 B — Mac (server) + 1 台 Pi (client)
需要把 yaml 里的 `num_clients` 改成 1 (或者用预置的 `mnist_pi_demo.yaml` 这种 1-client 配置).
```bash
# === 终端 1: Mac ===
source .venv/bin/activate
python -m fed_pi.server --config configs/<1client版.yaml>

# === 终端 2: 在 Mac 上 ssh 到 Pi 启动 client ===
ssh <PI1>
cd ~/fed_pi && source .venv/bin/activate
python -m fed_pi.client \
    --config configs/<1client版.yaml> \
    --cid 0 \
    --server_address <MAC_IP>:8081
```

### 模板 C — Mac (server) + 2 台 Pi (各 1 client)
yaml 里 `num_clients: 2`. 三个终端并行.
```bash
# === 终端 1: Mac ===
source .venv/bin/activate
python -m fed_pi.server --config configs/<实验名>.yaml

# === 终端 2: Pi-1 ===
ssh <PI1>
cd ~/fed_pi && source .venv/bin/activate
python -m fed_pi.client --config configs/<实验名>.yaml --cid 0 --server_address <MAC_IP>:8081

# === 终端 3: Pi-2 ===
ssh <PI2>
cd ~/fed_pi && source .venv/bin/activate
python -m fed_pi.client --config configs/<实验名>.yaml --cid 1 --server_address <MAC_IP>:8081
```

> 💡 **tmux 建议**: 在 Pi 上长任务用 `tmux new -s fl` 包一层, 即使 ssh 断开任务继续, 用 `tmux attach -t fl` 重新接管.

---

## 8 个实验逐个看

### E1 — MNIST + IID + 2 client + 5 轮 (基线)
- **目的**: 验证 FedAvg 全流程, 作为后续对照基线
- **预期**: 5 轮后准确率 ≈ 98%, 单机约 45 秒

```bash
# A) 本机模拟 (推荐先跑一次冒烟)
./scripts/run_local_simulation.sh configs/mnist_iid_2clients.yaml

# B) Mac + 1 Pi: 用 mnist_pi_demo.yaml (num_clients=1 已预置, 3 轮)
# Mac:
python -m fed_pi.server --config configs/mnist_pi_demo.yaml
# Pi:
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_pi_demo.yaml --cid 0 \
    --server_address <MAC_IP>:8081"

# C) Mac + 2 Pi:
# Mac:
python -m fed_pi.server --config configs/mnist_iid_2clients.yaml
# Pi-1:
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_iid_2clients.yaml --cid 0 \
    --server_address <MAC_IP>:8081"
# Pi-2:
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_iid_2clients.yaml --cid 1 \
    --server_address <MAC_IP>:8081"
```
**产出**: `results/mnist_iid_2clients/{metrics.json, curve_acc.png, curve_loss.png}`

---

### E2 — MNIST + Non-IID Dirichlet α=0.1 + 2 client + 15 轮 (极端 Non-IID)
- **目的**: 演示数据分布极不一致时 FedAvg 的退化
- **预期**: 准确率震荡, 15 轮后 80-90%, 比 IID 低 5-15 个点

```bash
# A) 本机模拟
./scripts/run_local_simulation.sh configs/mnist_noniid_a01.yaml

# B) Mac + 1 Pi: 需先把 yaml 复制改 num_clients=1 (Non-IID 在 1 client 下无意义, 不推荐)

# C) Mac + 2 Pi:
# Mac:
python -m fed_pi.server --config configs/mnist_noniid_a01.yaml
# Pi-1:
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_noniid_a01.yaml --cid 0 \
    --server_address <MAC_IP>:8081"
# Pi-2:
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_noniid_a01.yaml --cid 1 \
    --server_address <MAC_IP>:8081"
```
**产出**: `results/mnist_noniid_a01/...`

---

### E3 — MNIST + Non-IID α=0.5 + 2 client + 15 轮 (中等 Non-IID)
- **预期**: 收敛比 α=0.1 平稳, 准确率 ~92%

```bash
# A) 本机模拟
./scripts/run_local_simulation.sh configs/mnist_noniid_a05.yaml

# C) Mac + 2 Pi (推荐, 体现真机分布式)
python -m fed_pi.server --config configs/mnist_noniid_a05.yaml
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_noniid_a05.yaml --cid 0 --server_address <MAC_IP>:8081"
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_noniid_a05.yaml --cid 1 --server_address <MAC_IP>:8081"
```
**产出**: `results/mnist_noniid_a05/...`

---

### E4 — MNIST + Non-IID α=1.0 + 2 client + 15 轮 (轻度 Non-IID)
- **预期**: 接近 IID 基线表现, 准确率 ~97%

```bash
# A) 本机模拟
./scripts/run_local_simulation.sh configs/mnist_noniid_a10.yaml

# C) Mac + 2 Pi:
python -m fed_pi.server --config configs/mnist_noniid_a10.yaml
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_noniid_a10.yaml --cid 0 --server_address <MAC_IP>:8081"
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_noniid_a10.yaml --cid 1 --server_address <MAC_IP>:8081"
```

---

### E5 — MNIST + 样本量 9:1 不均衡 + 2 client + 10 轮
- **目的**: 演示 "大客户端 + 小客户端" 场景, FedAvg 加权后全局模型被大 client 主导
- **预期**: 准确率反映 client-0 (90% 数据) 的学习效果

```bash
# A) 本机模拟
./scripts/run_local_simulation.sh configs/mnist_imbalance_91.yaml

# C) Mac + 2 Pi:
python -m fed_pi.server --config configs/mnist_imbalance_91.yaml
# Pi-1 拿 90% 样本:
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_imbalance_91.yaml --cid 0 --server_address <MAC_IP>:8081"
# Pi-2 拿 10% 样本:
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && \
    python -m fed_pi.client --config configs/mnist_imbalance_91.yaml --cid 1 --server_address <MAC_IP>:8081"
```

---

### E6 — CIFAR-10 + 自定义 CNN + 2 client + 20 轮 (彩色图基线)
- **预期**: 20 轮后准确率 60-70%, 单 epoch 在 Pi 上 ~3-4 分钟

```bash
# A) 本机模拟 (推荐, Pi 上 20 轮太慢)
./scripts/run_local_simulation.sh configs/cifar10_cnn.yaml

# C) Mac + 2 Pi (耗时长, 约 2-3 小时):
python -m fed_pi.server --config configs/cifar10_cnn.yaml
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && tmux new -d -s fl \
    'python -m fed_pi.client --config configs/cifar10_cnn.yaml --cid 0 --server_address <MAC_IP>:8081'"
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && tmux new -d -s fl \
    'python -m fed_pi.client --config configs/cifar10_cnn.yaml --cid 1 --server_address <MAC_IP>:8081'"
# 查看 Pi 进度: ssh pi@pi-01.local "tmux attach -t fl"
```

---

### E7 — CIFAR-10 + MobileNetV2 + 2 client + 20 轮 (轻量化对比 1)
- **注意**: batch_size 已调小到 16, 避免 Pi 内存压力
- **预期**: 训练时间比 E6 长 (~50%), 内存高 (~700MB+), 准确率通常更好

```bash
# A) 本机模拟
./scripts/run_local_simulation.sh configs/cifar10_mobilenet.yaml

# C) Mac + 2 Pi:
python -m fed_pi.server --config configs/cifar10_mobilenet.yaml
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && tmux new -d -s fl \
    'python -m fed_pi.client --config configs/cifar10_mobilenet.yaml --cid 0 --server_address <MAC_IP>:8081'"
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && tmux new -d -s fl \
    'python -m fed_pi.client --config configs/cifar10_mobilenet.yaml --cid 1 --server_address <MAC_IP>:8081'"
```

---

### E8 — CIFAR-10 + SqueezeNet1_1 + 2 client + 20 轮 (轻量化对比 2)
- **预期**: 参数最少 (~0.7M), 内存占用最低, 训练时间介于 E6/E7 之间

```bash
# A) 本机模拟
./scripts/run_local_simulation.sh configs/cifar10_squeezenet.yaml

# C) Mac + 2 Pi:
python -m fed_pi.server --config configs/cifar10_squeezenet.yaml
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && tmux new -d -s fl \
    'python -m fed_pi.client --config configs/cifar10_squeezenet.yaml --cid 0 --server_address <MAC_IP>:8081'"
ssh pi@pi-02.local "cd ~/fed_pi && source .venv/bin/activate && tmux new -d -s fl \
    'python -m fed_pi.client --config configs/cifar10_squeezenet.yaml --cid 1 --server_address <MAC_IP>:8081'"
```

---

## 跑完后: 生成跨实验对比图

```bash
# 在 Mac 上 (实验产出都汇集在 Mac 的 results/ 下)
source .venv/bin/activate

# Non-IID 三档对比 (E2/E3/E4)
python scripts/plot_comparison.py --group noniid
# 产出: results/_comparison/{noniid_acc.png, noniid_loss.png}

# 轻量化网络对比 (E6/E7/E8)
python scripts/plot_comparison.py --group lightweight
# 产出: results/_comparison/{lightweight_acc.png, lightweight_resources.png, lightweight_summary.json}

# 一次性生成所有对比图
python scripts/plot_comparison.py --group all
```

---

## 一次性跑全部 8 个实验 (本机模拟)

```bash
./scripts/run_all_experiments.sh
# 自动按顺序跑 E1-E8, 最后生成对比图, 预计 30-60 分钟
```

---

## 常见问题

### Q1: server 和 client 启动顺序有要求吗?
**先 server, 再 client**. server 启动后等待 `num_clients` 个 client 全部连上才开始训练. client 连不到 server 会立即报错 `failed to connect`.

### Q2: 中途要中止怎么办?
- Mac server: `Ctrl+C` 直接退出, 已完成轮次的 metrics 不会丢失但当前轮结果不会保存
- Pi client: ssh 进去 `Ctrl+C` 退出 python 进程, 或者 `pkill -f fed_pi.client`
- tmux 后台进程: `ssh <PI> "tmux kill-session -t fl"`

### Q3: 我只有 1 台 Pi, 想跑 Non-IID 实验怎么办?
Non-IID 实验本质上需要 ≥ 2 个 client 才有意义 (一个 client 时数据全归它, 谈不上"分布不一致"). 退而求其次:
- 在 Mac 上跑 server + 1 个本地 client 进程, Pi 上跑另一个 client. 都用同一个 yaml (`num_clients: 2`).
- 命令同模板 C, 把 Pi-2 那一段换成在 Mac 上本地起 `python -m fed_pi.client ... --cid 1`

### Q4: 防火墙问题, Pi 上 `nc -zv <MAC_IP> 8081` 显示拒绝
- macOS: 系统设置 → 网络 → 防火墙 → 允许 Python 入站连接 (或临时关闭防火墙测试)
- Linux: `sudo ufw allow 8081/tcp`

### Q5: 怎么验证 Pi 上 venv 是否正确激活
```bash
ssh pi@pi-01.local "cd ~/fed_pi && source .venv/bin/activate && which python && python -c 'import flwr; print(flwr.__version__)'"
# 应输出 /home/pi/fed_pi/.venv/bin/python 和 flwr 版本号
```

### Q6: 第二台 Pi 怎么准备?
和 docs/setup_pi.md 一样的步骤. 也可以直接从 Pi-1 克隆 (省去重装依赖):
```bash
# 在 Pi-1 上
tar czf /tmp/fed_pi.tar.gz -C ~ fed_pi
scp /tmp/fed_pi.tar.gz pi@pi-02.local:/tmp/
ssh pi@pi-02.local "cd ~ && tar xzf /tmp/fed_pi.tar.gz"
```
注意: 跨 Pi 复制 .venv 要求两台 Pi 的 OS / Python 版本一致.

---

## 命令速查 (单页)

| 实验 | 本机模拟 | Mac+2Pi (Mac 端) | Pi 端 (cid=0) |
|------|---------|------------------|---------------|
| E1 mnist_iid | `./scripts/run_local_simulation.sh configs/mnist_iid_2clients.yaml` | `python -m fed_pi.server --config configs/mnist_iid_2clients.yaml` | `python -m fed_pi.client --config configs/mnist_iid_2clients.yaml --cid 0 --server_address <MAC_IP>:8081` |
| E2 noniid α=0.1 | `... configs/mnist_noniid_a01.yaml` | `... configs/mnist_noniid_a01.yaml` | `--config configs/mnist_noniid_a01.yaml --cid 0/1` |
| E3 noniid α=0.5 | `... configs/mnist_noniid_a05.yaml` | `... configs/mnist_noniid_a05.yaml` | `--config configs/mnist_noniid_a05.yaml --cid 0/1` |
| E4 noniid α=1.0 | `... configs/mnist_noniid_a10.yaml` | `... configs/mnist_noniid_a10.yaml` | `--config configs/mnist_noniid_a10.yaml --cid 0/1` |
| E5 imbalance 9:1 | `... configs/mnist_imbalance_91.yaml` | `... configs/mnist_imbalance_91.yaml` | `--config configs/mnist_imbalance_91.yaml --cid 0/1` |
| E6 cifar CNN | `... configs/cifar10_cnn.yaml` | `... configs/cifar10_cnn.yaml` | `--config configs/cifar10_cnn.yaml --cid 0/1` |
| E7 cifar MobileNet | `... configs/cifar10_mobilenet.yaml` | `... configs/cifar10_mobilenet.yaml` | `--config configs/cifar10_mobilenet.yaml --cid 0/1` |
| E8 cifar SqueezeNet | `... configs/cifar10_squeezenet.yaml` | `... configs/cifar10_squeezenet.yaml` | `--config configs/cifar10_squeezenet.yaml --cid 0/1` |
| Demo (1 Pi) | — | `python -m fed_pi.server --config configs/mnist_pi_demo.yaml` | `--config configs/mnist_pi_demo.yaml --cid 0` |
