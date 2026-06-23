# 架构说明

## FedAvg 流程

```
                ┌──────────────────────────────────┐
                │           Flower Server          │
                │  (Mac 或 Raspberry Pi, 0.0.0.0)  │
                │                                  │
                │  FedAvg 策略:                    │
                │   global_W = Σ (n_i/N) * W_i     │
                └──────────────┬───────────────────┘
                               │   gRPC (port 8081)
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────┐               ┌──────────────────┐
   │   Pi-1 Client    │               │   Pi-2 Client    │
   │   (cid=0)        │               │   (cid=1)        │
   │                  │               │                  │
   │  本地 MNIST 子集  │               │  本地 MNIST 子集  │
   │  (按 Dirichlet   │               │  (按 Dirichlet   │
   │   分到此 client) │               │   分到此 client) │
   │                  │               │                  │
   │  本地 SGD 1 epoch │               │  本地 SGD 1 epoch │
   │  ↓               │               │  ↓               │
   │  上传 W_0        │               │  上传 W_1        │
   └──────────────────┘               └──────────────────┘
```

### 每一轮 (round) 做什么

1. **server 选 client**: `fraction_fit=1.0`, 选全部已连接 client
2. **server → client**: 下发当前全局参数 `parameters`
3. **client**: `fit(parameters, config)`
   - `set_parameters(model, parameters)`  ← 加载全局参数
   - `local_train(model, ..., epochs=1)`  ← 本地 SGD
   - 返回 `(new_params, n_samples, metrics)`
4. **server 聚合**:
   - 参数: `global_W = Σ (n_i / N) * W_i`  (FedAvg 加权平均)
   - 指标: `weighted_avg_metrics()` 同样按样本数加权
5. **server → client**: 下发新的全局参数, 进入 `evaluate` 阶段
6. **client**: `evaluate(parameters, config)` 在本地测试集上算 loss / accuracy
7. **server 聚合 evaluate 指标**, 写入 History

`num_rounds` 轮结束后, server 返回 `History` 对象, 我们把它 dump 成
`results/<exp>/metrics.json` 并画曲线.

## 模块依赖图

```
client.py / server.py        ← 进程入口
        │
        ├──── utils.config    (YAML 加载与校验)
        ├──── utils.logger    (彩色日志)
        ├──── utils.sysmon    (CPU/内存监控)
        ├──── utils.metrics   (保存 JSON + 画图)  ← 仅 server
        ├──── data.datasets   (MNIST / CIFAR-10)
        ├──── data.partition  (IID/Dirichlet/Imbalance)
        ├──── models.factory  (4 个模型)
        └──── train.params    (state_dict ↔ list[np.ndarray])
              train.trainer   (local_train / evaluate)
```

## 数据划分策略

| 策略 | 函数 | 关键参数 | 适用场景 |
|------|------|----------|----------|
| IID | `partition_iid` | (无) | 理想基线; 各 client 数据同分布 |
| Dirichlet Non-IID | `partition_dirichlet` | `alpha` | 模拟客户端数据异构 |
| Imbalance | `partition_imbalance` | `ratios` | 模拟客户端样本数极度不均 |

Dirichlet 的 `alpha` 含义:
- `α → 0`: 每类样本几乎全分给一个 client (极端 Non-IID)
- `α = 1`: 比较随机
- `α → ∞`: 趋近均匀 (接近 IID)

实验中常用 0.1 / 0.5 / 1.0 三档对比.

## 参数序列化的设计选择

我们用 `model.state_dict()` 而不是 `model.parameters()`:
- `state_dict` 包含 BatchNorm 的 `running_mean` / `running_var` (buffer, 非可训练参数)
- 但这些 buffer 对推理至关重要 — 不同步过去会导致全局模型 BN 失效
- 顺序由 `state_dict.keys()` 决定, get / set 必须严格对齐

代价: 通信量比纯 trainable params 略大. 课程场景下完全可接受.

## 监控指标

每个 client 在 `fit` / `evaluate` 阶段上报:

| 指标 | 来源 | 含义 |
|------|------|------|
| `train_loss` | `local_train` 返回 | 最后一个 epoch 平均交叉熵 |
| `train_time` | `SysMonitor.wall_time` | fit 阶段墙钟时间 (秒) |
| `cpu_avg` | `SysMonitor` | fit 阶段单核等价 CPU 占用 (%) |
| `mem_peak_mb` | `SysMonitor` | fit 阶段进程 RSS 内存峰值 (MB) |
| `accuracy` | `evaluate` 返回 | 测试集分类准确率 |

server 端 `weighted_avg_metrics` 按样本数加权平均, 写入 `metrics.json`.

## 范围外

为了让代码贴近教学而不发散, 以下东西**没**实现:

- 安全聚合 (SecAgg) / 差分隐私 (DP)
- 异步联邦 (FedAsync) / 其他聚合算法 (FedProx, SCAFFOLD, ...)
- GPU 训练 (Pi 无 CUDA, 统一 CPU)
- Web UI 监控 (日志 + PNG 足矣)
- 多机 server (单 server 即可)

需要扩展时, 模型/数据/划分都已经是工厂模式, 新增只需 register 一行.
