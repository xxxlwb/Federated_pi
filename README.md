# fed_pi — 树莓派联邦学习系统

> 课程作业项目: 在 1~2 台树莓派 (或本机) 上跑通基于 **PyTorch + Flower** 的联邦学习,
> 覆盖 MNIST/CIFAR-10、IID/Non-IID/不均衡、CNN/MobileNetV2/SqueezeNet 共 8 个实验.

## 项目结构

```
fed_pi/
├── fed_pi/              # 主代码包
│   ├── client.py        # FlowerClient + 入口
│   ├── server.py        # FedAvg 策略 + 入口 + 保存指标与曲线
│   ├── data/            # 数据集 + 三种联邦划分 (IID/Dirichlet/Imbalance)
│   ├── models/          # 模型工厂 (MnistCNN, Cifar10CNN, MobileNetV2, SqueezeNet1_1)
│   ├── train/           # 本地训练 + 参数 list[np.ndarray] 互转
│   └── utils/           # 配置 / 日志 / 系统监控 / 指标可视化
├── configs/             # 8 个实验 YAML
├── scripts/             # 运行脚本 (单机/真机/批量/对比图)
├── docs/                # 树莓派部署文档 + 架构说明
├── tests/               # pytest 单测
├── requirements.txt     # Mac/Linux 依赖
└── requirements-pi.txt  # 树莓派依赖 (用 piwheels)
```

## 快速开始 (Mac/Linux 本地模拟)

```bash
cd fed_pi

# 1. 建虚拟环境
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. 跑单元测试 (划分/模型 forward)
pytest tests/ -v

# 3. 跑最小实验 (E1 基线: MNIST + IID + 2 client + 5 轮)
./scripts/run_local_simulation.sh configs/mnist_iid_2clients.yaml

# 产出:
#   results/mnist_iid_2clients/metrics.json
#   results/mnist_iid_2clients/curve_acc.png
#   results/mnist_iid_2clients/curve_loss.png
```

## 真机部署 (1~2 台树莓派)

详见 `docs/setup_pi.md`(环境搭建)和 **`docs/run_experiments.md`(逐个实验的 Mac/Pi 端命令清单)**。简略步骤:

1. 树莓派 Pi OS 64-bit, Python 3.11, 开 2GB swap
2. `pip install -r requirements-pi.txt --extra-index-url https://www.piwheels.org/simple`
3. Mac 启动 server: `./scripts/run_server.sh configs/mnist_iid_2clients.yaml`
4. Pi 启动 client: `./scripts/run_client.sh configs/mnist_iid_2clients.yaml 0 <mac_ip>:8081`
   (如果有第二台 Pi 就再启动一个 cid=1)

> 📖 **每个实验在 Mac 和 Pi 上的精确命令对照**(本机模拟 / 1 Pi / 2 Pi 三种模式),
> 见 [`docs/run_experiments.md`](docs/run_experiments.md)。

## 实验设计

| 编号 | 配置 | 轮数 | 重点观察 |
|------|------|------|----------|
| E1 | `mnist_iid_2clients` | 5 | 基线 ≈ 97% acc |
| E2 | `mnist_noniid_a01` | 15 | Dirichlet α=0.1, 极端 Non-IID |
| E3 | `mnist_noniid_a05` | 15 | α=0.5 |
| E4 | `mnist_noniid_a10` | 15 | α=1.0, 接近 IID |
| E5 | `mnist_imbalance_91` | 10 | 样本量 9:1 不均衡 |
| E6 | `cifar10_cnn` | 20 | CIFAR-10 + 自定义 CNN, 基线 |
| E7 | `cifar10_mobilenet` | 20 | MobileNetV2 对比 |
| E8 | `cifar10_squeezenet` | 20 | SqueezeNet1_1 对比 |

### 批量跑

```bash
./scripts/run_all_experiments.sh         # 串行跑 8 个
python scripts/plot_comparison.py --group noniid       # E2/E3/E4 同框
python scripts/plot_comparison.py --group lightweight  # E6/E7/E8 同框
```

## 配置文件结构

```yaml
exp_name: mnist_noniid_a05
dataset: mnist                  # mnist | cifar10
model: mnist_cnn                # mnist_cnn | cifar_cnn | mobilenet_v2 | squeezenet1_1
num_classes: 10
num_clients: 2                  # 1 或 2 均可
num_rounds: 10
local_epochs: 1
batch_size: 32
learning_rate: 0.01
partition:
  strategy: dirichlet           # iid | dirichlet | imbalance
  alpha: 0.5                    # dirichlet 用
  ratios: [0.9, 0.1]            # imbalance 用
server_address: "0.0.0.0:8081"
seed: 42
device: cpu
```

## 关键设计

- **配置驱动**: 切换实验只改 YAML, 不动代码
- **支持 1 台或 2 台树莓派**: 改 `num_clients`, server 的 `min_available_clients` 自动跟随
- **两种运行模式并存**:
  - 单机多进程 (开发调试): `run_local_simulation.sh`
  - 真实分布式 (树莓派): server / client 分别启动, client 指定 `--server_address`
- **模型 / 数据 / 划分解耦**: 都是工厂函数, 新增只需 register 一行

## 监控指标

每一轮 server 会从所有 client 收集 (经样本数加权平均):

- **fit 阶段**: train_loss, train_time, cpu_avg, mem_peak_mb
- **evaluate 阶段**: loss, accuracy

存到 `results/<exp>/metrics.json`, 并自动绘制 `curve_acc.png` / `curve_loss.png`.

## 进一步阅读

- [`docs/run_experiments.md`](docs/run_experiments.md) — **8 个实验在 Mac / 1 Pi / 2 Pi 上的逐字命令清单**(最常用)
- [`docs/setup_pi.md`](docs/setup_pi.md) — 树莓派环境搭建详细步骤
- [`docs/architecture.md`](docs/architecture.md) — FedAvg 流程图与模块依赖
