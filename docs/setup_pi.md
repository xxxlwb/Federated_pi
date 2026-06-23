# 树莓派环境搭建

适用: Raspberry Pi 4 / Raspberry Pi 5, Pi OS 64-bit (Bookworm).

## 0. 检查系统

```bash
uname -m         # 应输出 aarch64
python3 --version # Bookworm 默认 3.11
free -h          # 看可用内存
```

## 1. 系统准备

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git tmux libopenblas-dev
```

`libopenblas-dev` 是 PyTorch 的运行时依赖, 不装会报 `cannot open shared object file`.

## 2. 开 2GB swap (必做)

Pi 4/5 默认只有 100MB swap, 跑 MobileNetV2 + CIFAR-10 内存峰值会触发 OOM. 调到 2GB:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
free -h    # 应看到 Swap 2.0G
```

## 3. 克隆项目并建 venv

```bash
git clone <你的仓库或 scp 复制项目> ~/fed_pi
cd ~/fed_pi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 4. 安装 PyTorch + Flower

官方 PyPI 不提供 aarch64 wheel, 走 piwheels:

```bash
pip install -r requirements-pi.txt \
    --extra-index-url https://www.piwheels.org/simple
```

> 如果上面装不上, 降级到旧版:
> ```bash
> pip install torch==2.0.1 torchvision==0.15.2 \
>     --index-url https://www.piwheels.org/simple \
>     --extra-index-url https://pypi.org/simple
> pip install "flwr>=1.16,<1.32" matplotlib psutil PyYAML
> ```

验证:

```bash
python -c "import torch, flwr; print(torch.__version__, flwr.__version__)"
```

## 5. 网络互通测试

假设 server 在 Mac (IP `192.168.1.100`), client 在 Pi (IP `192.168.1.11`).

**Pi 上**:
```bash
ping -c 3 192.168.1.100
nc -zv 192.168.1.100 8081    # 启动 server 后再测; 应输出 "Connection ... succeeded"
```

**Mac 上** (server):
```bash
# 防火墙临时放行
# macOS 系统设置 → 网络 → 防火墙 → 关闭或允许 python
ifconfig | grep "inet "      # 查 IP
./scripts/run_server.sh configs/mnist_iid_2clients.yaml
```

server 启动后会绑定 `0.0.0.0:8081`, 等待 client 连接.

## 6. 启动 client

**两台 Pi 各开一个 tmux 会话** (推荐, 实验过程可现场展示日志):

```bash
# Pi-1 (192.168.1.11)
ssh pi@192.168.1.11
cd ~/fed_pi
source .venv/bin/activate
tmux new -s fl
./scripts/run_client.sh configs/mnist_iid_2clients.yaml 0 192.168.1.100:8081
# Ctrl-B D 退出 tmux 但保留进程

# Pi-2 (192.168.1.12)
ssh pi@192.168.1.12
cd ~/fed_pi && source .venv/bin/activate
tmux new -s fl
./scripts/run_client.sh configs/mnist_iid_2clients.yaml 1 192.168.1.100:8081
```

## 7. 只有 1 台树莓派怎么办

把 cfg 里 `num_clients` 改成 1 即可, server 的 `min_available_clients=1`,
单 client 也能跑通 FedAvg 流程 (退化成纯本地训练 + 周期性参数同步).

```bash
# 临时改 cfg (或新建一个 _1client.yaml)
sed 's/num_clients: 2/num_clients: 1/' configs/mnist_iid_2clients.yaml > /tmp/single.yaml
# 然后在 Pi 上启动 server + client (注意, 1 个 Pi 同时跑 server+client 内存吃紧, 推荐 Mac 跑 server)
```

## 8. 实验中的注意点

- **batch_size 调小**: MNIST 用 32 没问题; CIFAR + MobileNetV2 建议降到 8;
  CIFAR + Cifar10CNN 16~32 都可
- **温度**: 长时间训练后 `vcgencmd measure_temp`, 超过 80°C 会触发降频
- **断网恢复**: Flower 默认 grpc 连接断了不会自动重连; client 进程退出即可,
  Pi 上 `tmux attach -t fl` 检查
- **clean 重跑**: `rm -rf results/<exp_name>` 重跑某个实验

## 9. 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `pip install torch` 失败, no matching distribution | 走的 PyPI 没 aarch64 轮子 | 加 `--extra-index-url https://www.piwheels.org/simple` |
| `OSError: libopenblas.so.0: cannot open` | 缺系统库 | `sudo apt install libopenblas-dev` |
| client 连不上 server | server 没监听 `0.0.0.0`, 或防火墙 | 检查 cfg 的 server_address; `nc -zv` 验证 |
| OOM killed | swap 太小或 batch 太大 | 开 2GB swap, 降 batch_size |
| 训练比预期慢很多 | 单核 100% 跑满, Pi 4 算力本身就有限 | 实属正常, 5 轮 MNIST 约 1-3 分钟 |
