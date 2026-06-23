"""配置文件加载.

约定:
- 所有实验配置放在 configs/*.yaml
- load_config 会校验必填字段, 缺失抛 ValueError, 便于早失败
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 必填字段; 任意一项缺失即抛错
_REQUIRED_KEYS = [
    "exp_name",
    "dataset",
    "model",
    "num_classes",
    "num_clients",
    "num_rounds",
    "local_epochs",
    "batch_size",
    "learning_rate",
    "partition",
    "server_address",
    "seed",
    "device",
]


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并做轻量校验.

    Args:
        path: 配置文件路径

    Returns:
        dict 形式的配置

    Raises:
        FileNotFoundError: 路径不存在
        ValueError: 必填字段缺失或 partition.strategy 非法
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config {p.name} missing keys: {missing}")

    strat = cfg["partition"].get("strategy")
    if strat not in {"iid", "dirichlet", "imbalance"}:
        raise ValueError(
            f"partition.strategy must be one of iid/dirichlet/imbalance, got {strat!r}"
        )

    return cfg
