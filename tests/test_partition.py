"""数据划分单元测试.

不依赖真实数据集 (避免 CI/初次跑慢), 用 numpy 假数据集.
"""
from __future__ import annotations

import numpy as np
import pytest

from fed_pi.data.partition import (
    get_client_indices,
    partition_dirichlet,
    partition_iid,
    partition_imbalance,
)


class FakeDataset:
    """模拟 torchvision dataset: 暴露 .targets 与 __len__."""

    def __init__(self, n: int = 1000, num_classes: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.targets = rng.integers(0, num_classes, size=n).tolist()

    def __len__(self) -> int:
        return len(self.targets)


# ---------- IID ----------

def test_partition_iid_covers_all_samples():
    ds = FakeDataset(1000)
    parts = partition_iid(ds, num_clients=4, seed=42)
    assert len(parts) == 4
    all_idx = sum(parts, [])
    assert sorted(all_idx) == list(range(1000))
    # 各 client 数量大致相等
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1


def test_partition_iid_deterministic():
    ds = FakeDataset(500)
    a = partition_iid(ds, 3, seed=42)
    b = partition_iid(ds, 3, seed=42)
    assert a == b


def test_partition_iid_single_client():
    """退化场景: num_clients=1 时, client 0 应拿到全部样本."""
    ds = FakeDataset(100)
    parts = partition_iid(ds, num_clients=1, seed=42)
    assert len(parts) == 1
    assert sorted(parts[0]) == list(range(100))


# ---------- Dirichlet ----------

def test_partition_dirichlet_covers_all_samples():
    ds = FakeDataset(1000, num_classes=10)
    parts = partition_dirichlet(ds, num_clients=2, alpha=0.5, seed=42)
    all_idx = sorted(sum(parts, []))
    assert all_idx == list(range(1000))


def test_partition_dirichlet_alpha_skew():
    """alpha 越小 → 客户端越偏: 客户端持有类别数应更少."""
    ds = FakeDataset(2000, num_classes=10)
    targets = np.array(ds.targets)

    def avg_classes_per_client(parts):
        return np.mean([len(set(targets[p].tolist())) for p in parts])

    extreme = partition_dirichlet(ds, num_clients=2, alpha=0.05, seed=42)
    moderate = partition_dirichlet(ds, num_clients=2, alpha=10.0, seed=42)

    # alpha 大 → 每 client 平均持有的类别数 应 >= alpha 小的情况
    assert avg_classes_per_client(extreme) <= avg_classes_per_client(moderate)


# ---------- Imbalance ----------

def test_partition_imbalance_ratio_respected():
    ds = FakeDataset(1000)
    parts = partition_imbalance(ds, num_clients=2, ratios=[0.9, 0.1], seed=42)
    assert len(parts[0]) >= 850 and len(parts[0]) <= 950
    assert len(parts[1]) >= 50 and len(parts[1]) <= 150
    assert sorted(sum(parts, [])) == list(range(1000))


def test_partition_imbalance_validates_ratios():
    ds = FakeDataset(100)
    with pytest.raises(ValueError):
        partition_imbalance(ds, num_clients=2, ratios=[0.5, 0.6])  # sum != 1
    with pytest.raises(ValueError):
        partition_imbalance(ds, num_clients=2, ratios=[1.0])  # 长度不匹配


# ---------- get_client_indices 统一入口 ----------

def test_get_client_indices_iid():
    ds = FakeDataset(200)
    cfg = {
        "num_clients": 2,
        "seed": 42,
        "partition": {"strategy": "iid"},
    }
    idx0 = get_client_indices(ds, cfg, 0)
    idx1 = get_client_indices(ds, cfg, 1)
    assert len(idx0) + len(idx1) == 200
    assert set(idx0).isdisjoint(set(idx1))


def test_get_client_indices_validates_cid():
    ds = FakeDataset(100)
    cfg = {"num_clients": 2, "seed": 42, "partition": {"strategy": "iid"}}
    with pytest.raises(ValueError):
        get_client_indices(ds, cfg, 5)


def test_get_client_indices_dirichlet_needs_alpha():
    ds = FakeDataset(100)
    cfg = {"num_clients": 2, "seed": 42, "partition": {"strategy": "dirichlet"}}
    with pytest.raises(ValueError, match="alpha"):
        get_client_indices(ds, cfg, 0)
