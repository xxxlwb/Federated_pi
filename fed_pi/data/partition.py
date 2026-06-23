"""联邦学习的三种数据划分策略.

- partition_iid: 均匀打乱后平均分 (理想情况, 各 client 数据同分布)
- partition_dirichlet: 每类按 Dirichlet([alpha]*n) 切分 (Non-IID), alpha 越小越偏
- partition_imbalance: 按指定比例切总样本数 (模拟弱客户端样本少)

所有函数返回 `list[list[int]]`, 第 i 个子列表是 client i 的样本索引.
配合 torch.utils.data.Subset(trainset, indices) 即可得到本地数据集.
"""
from __future__ import annotations

import numpy as np

from fed_pi.data.datasets import get_targets


def partition_iid(trainset, num_clients: int, seed: int = 42) -> list[list[int]]:
    """完全随机均分.

    用 numpy 的 PCG64 生成器, 同一 seed 在所有 client 上得到一致的划分,
    这一点很重要 — 因为每个 client 进程会独立调用此函数, 必须能算出同样的索引.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trainset))
    return [a.tolist() for a in np.array_split(idx, num_clients)]


def partition_dirichlet(
    trainset, num_clients: int, alpha: float, seed: int = 42
) -> list[list[int]]:
    """按 Dirichlet 分布划分每个类别的样本.

    实现思路 (经典做法, 参考 Yurochkin et al. 2019 / Hsu et al. 2019):
    对每个类别 c:
        1. 取出该类所有样本索引
        2. 用 Dirichlet([alpha]*num_clients) 采样一组比例 p = (p_0, ..., p_{n-1})
        3. 按 p 把这些索引切分到各 client
    alpha 含义:
        alpha → 0 : 每类几乎全分给一个 client (极端 Non-IID)
        alpha = 1 : 比较随机
        alpha → ∞ : 趋近均匀 (接近 IID)
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(get_targets(trainset))
    num_classes = int(labels.max()) + 1

    client_idx: list[list[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * num_clients)
        # 切分点 (前 n-1 个), np.split 会自动取剩余给最后一块
        split_pts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx_c, split_pts)):
            client_idx[cid].extend(chunk.tolist())
    return client_idx


def partition_imbalance(
    trainset, num_clients: int, ratios: list[float], seed: int = 42
) -> list[list[int]]:
    """按比例分配样本数, 模拟客户端样本极度不均衡.

    Args:
        ratios: 长度等于 num_clients, 和必须为 1.0. 例: [0.9, 0.1] 表示 9:1.
    """
    if len(ratios) != num_clients:
        raise ValueError(f"ratios length {len(ratios)} != num_clients {num_clients}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trainset))
    split_pts = (np.cumsum(ratios) * len(idx)).astype(int)[:-1]
    return [a.tolist() for a in np.split(idx, split_pts)]


def get_client_indices(trainset, cfg: dict, client_id: int) -> list[int]:
    """统一入口: 根据 cfg["partition"]["strategy"] 派发到对应函数."""
    if client_id < 0 or client_id >= cfg["num_clients"]:
        raise ValueError(
            f"client_id {client_id} out of range [0, {cfg['num_clients']})"
        )

    p = cfg["partition"]
    n = cfg["num_clients"]
    seed = cfg["seed"]
    strat = p["strategy"]

    if strat == "iid":
        parts = partition_iid(trainset, n, seed)
    elif strat == "dirichlet":
        if "alpha" not in p:
            raise ValueError("dirichlet partition requires partition.alpha in config")
        parts = partition_dirichlet(trainset, n, float(p["alpha"]), seed)
    elif strat == "imbalance":
        if "ratios" not in p:
            raise ValueError("imbalance partition requires partition.ratios in config")
        parts = partition_imbalance(trainset, n, list(p["ratios"]), seed)
    else:
        raise ValueError(f"unknown partition strategy: {strat}")
    return parts[client_id]
